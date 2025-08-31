import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from scipy.sparse import hstack
import json
import re
import warnings
import pickle

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

# ---------- helpers ----------
def safe_json_list_parser(x):
    """Safely parse number/comma-separated/JSON list strings to list; fallback []."""
    try:
        s = str(x).strip()
        if not s.startswith('['):
            s = f"[{s}]"
        return json.loads(s.replace("'", "\""))
    except:
        return []

def ensure_cols(df, cols, fill=0):
    """Ensure required columns exist in df (adding with fill if missing)."""
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df
# -----------------------------

class AltiCredScorer:
    """
    Stacked model producing an AltiCred score for salaried employees.
    """
    def __init__(self, file_path='data/salaried_dataset.csv'):
        self.file_path = file_path
        self.df = self._load_and_clean_data()
        self.df_columns = self.df.columns.tolist()  # <-- make column schema available everywhere

        # Sentiment
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Train
        self._train_all_models()
        self._train_autoencoder()

    def _load_and_clean_data(self):
        try:
            df = pd.read_csv(self.file_path)
            print("Dataset loaded successfully.")
            df.ffill(inplace=True)
            return df
        except FileNotFoundError:
            print(f"Error: The file was not found at {self.file_path}")
            exit()

    def _train_all_models(self):
        print("\n--- Training All Base Models ---")
        self._train_digital_trust_model()
        self._train_resilience_model()
        self._train_adaptability_model()
        self._train_language_sentiment_model()
        self._train_meta_model()
        print("\n--- All Models Trained Successfully ---")

    def _train_digital_trust_model(self):
        df_trust = self.df.dropna(subset=['defaulter_neighbors', 'verified_neighbors', 'connections', 'default_label']).copy()
        df_trust['num_connections'] = df_trust['connections'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

        features = ['defaulter_neighbors', 'verified_neighbors', 'num_connections']
        X = df_trust[features]
        y = df_trust['default_label']

        self.trust_scaler = StandardScaler().fit(X)
        X_scaled = self.trust_scaler.transform(X)

        input_dim = X_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoder_layer = Dense(2, activation='relu')(input_layer)
        self.trust_encoder = Model(inputs=input_layer, outputs=encoder_layer)

        # NOTE: This encoder is untrained; it's used as a simple projection.
        X_encoded = self.trust_encoder.predict(X_scaled, verbose=0)

        ratio = y.value_counts()[0] / y.value_counts()[1]
        self.trust_model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=ratio
        )
        self.trust_model.fit(X_encoded, y)
        print("1. Digital Trust Model Trained.")

    def _train_resilience_model(self):
        needs = ['monthly_credit_bills','bnpl_utilization_rate','mortgage_months_left',
                 'income-expense ratio','upi_balances','emi_status_log','default_label']
        df_resilience = self.df.dropna(subset=needs).copy()
        df_resilience['avg_upi_balance'] = df_resilience['upi_balances'].apply(
            lambda x: np.mean(safe_json_list_parser(x)) if safe_json_list_parser(x) else 0)
        df_resilience['missed_emi_count'] = df_resilience['emi_status_log'].apply(
            lambda x: safe_json_list_parser(x).count(0))

        self.resilience_features = [
            'monthly_credit_bills', 'bnpl_utilization_rate', 'mortgage_months_left',
            'avg_upi_balance', 'income-expense ratio', 'missed_emi_count'
        ]
        X = df_resilience[self.resilience_features]
        y = df_resilience['default_label']

        self.resilience_scaler = StandardScaler().fit(X)
        X_scaled = self.resilience_scaler.transform(X)

        self.resilience_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        self.resilience_model.fit(X_scaled, y)
        print("2. Resilience Model Trained.")

    def _train_adaptability_model(self):
        needs = ['owns_home','monthly_rent','income-expense ratio','emi_status_log',
                 'recovery_days','monthly_credit_bills','mortgage_status','default_label']
        df_adapt = self.df.dropna(subset=needs).copy()
        df_adapt['missed_emi_count'] = df_adapt['emi_status_log'].apply(
            lambda x: safe_json_list_parser(x).count(0))
        mortgage_dummies = pd.get_dummies(df_adapt['mortgage_status'], prefix='mortgage')
        self.adapt_mortgage_cols = mortgage_dummies.columns
        df_adapt = pd.concat([df_adapt, mortgage_dummies], axis=1)

        self.adapt_features = [
            'owns_home','monthly_rent','income-expense ratio','missed_emi_count',
            'recovery_days','monthly_credit_bills'
        ] + list(self.adapt_mortgage_cols)

        X = df_adapt[self.adapt_features]
        y = df_adapt['default_label']

        self.adapt_scaler = StandardScaler().fit(X)
        X_scaled = self.adapt_scaler.transform(X)

        rf = RandomForestClassifier(random_state=42, class_weight='balanced').fit(X_scaled, y)
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced').fit(X_scaled, y)
        self.adapt_model = rf if accuracy_score(y, rf.predict(X_scaled)) > accuracy_score(y, lr.predict(X_scaled)) else lr
        print("3. Adaptability Model Trained.")

    def _train_language_sentiment_model(self):
        df_lang = self.df.dropna(subset=['user_posts', 'default_label']).copy()
        df_lang['cleaned_posts'] = df_lang['user_posts'].apply(
            lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))

        get_score = lambda text: self.sentiment_analyzer.polarity_scores(text)['compound']
        df_lang['generated_sentiment_score'] = df_lang['user_posts'].apply(get_score)

        self.lang_vectorizer = TfidfVectorizer(max_features=100, stop_words='english').fit(df_lang['cleaned_posts'])
        X_text = self.lang_vectorizer.transform(df_lang['cleaned_posts'])
        X_sentiment = df_lang['generated_sentiment_score'].values.reshape(-1, 1)

        X_combined = hstack([X_text, X_sentiment])
        y = df_lang['default_label']

        self.lang_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced').fit(X_combined, y)
        print("4. Language Sentiment Model Trained.")

    def _train_meta_model(self):
        meta_features = pd.DataFrame(index=self.df.index)
        meta_features['trust_score'] = self._predict_digital_trust_score(self.df)
        meta_features['resilience_score'] = self._predict_resilience_score(self.df)
        meta_features['adaptability_score'] = self._predict_adaptability_score(self.df)
        meta_features['language_score'] = self._predict_language_sentiment_score(self.df)
        meta_features.fillna(meta_features.median(), inplace=True)

        self.meta_model = LogisticRegression(random_state=42, class_weight='balanced').fit(
            meta_features, self.df['default_label'])
        print("5. Meta-Model Trained.")

    def _train_autoencoder(self):
        print("\n--- Training Autoencoder for Data Generation ---")
        autoencoder_features = self.df.select_dtypes(include=np.number).columns.tolist()
        df_ae = self.df[autoencoder_features].copy()
        df_ae.fillna(df_ae.median(), inplace=True)

        self.ae_scaler = MinMaxScaler().fit(df_ae)
        X_scaled = self.ae_scaler.transform(df_ae)

        input_dim = X_scaled.shape[1]
        encoding_dim = max(2, int(input_dim / 2))

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

        self.autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)

        self.encoder = Model(input_layer, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        print("Autoencoder Trained.")

    def generate_synthetic_data(self, n_samples=100):
        print(f"\n--- Generating {n_samples} Synthetic Data Points ---")
        latent_dim = self.encoder.input_shape[1]
        random_latent_vectors = np.random.normal(size=(n_samples, latent_dim))
        generated_data_scaled = self.decoder.predict(random_latent_vectors, verbose=0)
        generated_df = self.ae_scaler.inverse_transform(generated_data_scaled)
        generated_df = pd.DataFrame(generated_df, columns=self.df.select_dtypes(include=np.number).columns)
        print("Synthetic data generated.")
        return generated_df

    # -------------------- PREDICTION UTILITIES (no self.df dependency) --------------------

    def _predict_digital_trust_score(self, data):
        df = data.copy()
        df = ensure_cols(df, ['connections', 'defaulter_neighbors', 'verified_neighbors'], fill=0)
        df['num_connections'] = df['connections'].apply(lambda x: len(str(x).split(',')) if isinstance(x, str) else 0)
        X = df[['defaulter_neighbors', 'verified_neighbors', 'num_connections']]
        X_scaled = self.trust_scaler.transform(X)
        X_encoded = self.trust_encoder.predict(X_scaled, verbose=0)
        return self.trust_model.predict_proba(X_encoded)[:, 1]

    def _predict_resilience_score(self, data):
        df = data.copy()
        df = ensure_cols(df, ['monthly_credit_bills','bnpl_utilization_rate','mortgage_months_left',
                              'income-expense ratio','upi_balances','emi_status_log'], fill=0)
        df['avg_upi_balance'] = df['upi_balances'].apply(
            lambda x: np.mean(safe_json_list_parser(x)) if safe_json_list_parser(x) else 0)
        df['missed_emi_count'] = df['emi_status_log'].apply(
            lambda x: safe_json_list_parser(x).count(0))
        X = df[self.resilience_features]
        X_scaled = self.resilience_scaler.transform(X)
        return self.resilience_model.predict_proba(X_scaled)[:, 1]

    def _predict_adaptability_score(self, data):
        df = data.copy()
        df = ensure_cols(df, ['owns_home','monthly_rent','income-expense ratio',
                              'emi_status_log','recovery_days','monthly_credit_bills','mortgage_status'], fill=0)
        # if mortgage_status missing/0, make string for get_dummies to work
        if 'mortgage_status' in df.columns:
            df['mortgage_status'] = df['mortgage_status'].replace(0, 'unknown').astype(str)
        else:
            df['mortgage_status'] = 'unknown'

        df['missed_emi_count'] = df['emi_status_log'].apply(lambda x: safe_json_list_parser(x).count(0))
        mortgage_dummies = pd.get_dummies(df['mortgage_status'], prefix='mortgage')
        df = pd.concat([df, mortgage_dummies], axis=1)

        # ensure all trained dummy cols exist
        for col in self.adapt_mortgage_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[self.adapt_features]
        X_scaled = self.adapt_scaler.transform(X)
        return self.adapt_model.predict_proba(X_scaled)[:, 1]

    def _predict_language_sentiment_score(self, data):
        df = data.copy()
        df = ensure_cols(df, ['user_posts'], fill="")
        df['cleaned_posts'] = df['user_posts'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
        get_score = lambda text: self.sentiment_analyzer.polarity_scores(text)['compound']
        df['generated_sentiment_score'] = df['user_posts'].apply(get_score)

        X_text = self.lang_vectorizer.transform(df['cleaned_posts'])
        X_sentiment = df['generated_sentiment_score'].values.reshape(-1, 1)
        X_combined = hstack([X_text, X_sentiment])
        return self.lang_model.predict_proba(X_combined)[:, 1]

    # --------------------------------------------------------------------------------------

    def predict_alticred_score(self, user_data):
        """Return final AltiCred score in [0,1] for a single user dict."""
        user_df = pd.DataFrame([user_data])

        s1 = self._predict_digital_trust_score(user_df)[0]
        s2 = self._predict_resilience_score(user_df)[0]
        s3 = self._predict_adaptability_score(user_df)[0]
        s4 = self._predict_language_sentiment_score(user_df)[0]

        base_scores = np.array([[s1, s2, s3, s4]])
        meta_prediction_proba = self.meta_model.predict_proba(base_scores)[0][1]

        risks = base_scores.flatten()
        total_risk = np.sum(risks)
        weights = risks / total_risk if total_risk > 0 else np.array([0.25, 0.25, 0.25, 0.25])
        weighted_score = np.sum(risks * weights)

        risk_score = (0.5 * meta_prediction_proba) + (0.5 * weighted_score)
        final_score = 1 - risk_score
        return np.clip(final_score, 0, 1)

    def evaluate_model(self):
        """Evaluate on the training data (only available on trained instance)."""
        if getattr(self, 'df', None) is None:
            raise ValueError("Training DataFrame not available on a loaded model. Train first or evaluate during training run.")

        print("\n--- Evaluating Meta-Model Performance ---")
        meta_features = pd.DataFrame(index=self.df.index)
        meta_features['trust_score'] = self._predict_digital_trust_score(self.df)
        meta_features['resilience_score'] = self._predict_resilience_score(self.df)
        meta_features['adaptability_score'] = self._predict_adaptability_score(self.df)
        meta_features['language_score'] = self._predict_language_sentiment_score(self.df)
        meta_features.fillna(meta_features.median(), inplace=True)

        y_true = self.df['default_label']
        y_pred = self.meta_model.predict(meta_features)
        y_proba = self.meta_model.predict_proba(meta_features)[:, 1]

        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        rmse = np.sqrt(mean_squared_error(y_true, y_proba))
        r2 = r2_score(y_true, y_proba)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print("-----------------------------------------")

    def save_model(self, filepath='models/alticred_salaried_model.pkl'):
        """Save trained components to a pickle."""
        model_data = {
            'trust_scaler': self.trust_scaler,
            'trust_encoder': self.trust_encoder,
            'trust_model': self.trust_model,
            'resilience_scaler': self.resilience_scaler,
            'resilience_features': self.resilience_features,
            'resilience_model': self.resilience_model,
            'adapt_scaler': self.adapt_scaler,
            'adapt_features': self.adapt_features,
            'adapt_mortgage_cols': self.adapt_mortgage_cols,
            'adapt_model': self.adapt_model,
            'lang_vectorizer': self.lang_vectorizer,
            'lang_model': self.lang_model,
            'meta_model': self.meta_model,
            'ae_scaler': self.ae_scaler,
            'autoencoder': self.autoencoder,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'sentiment_analyzer': self.sentiment_analyzer,
            'df_columns': self.df_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath='models/alticred_salaried_model.pkl'):
        """Load a trained model from a pickle (no training data included)."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls.__new__(cls)

        # components
        instance.trust_scaler = model_data['trust_scaler']
        instance.trust_encoder = model_data['trust_encoder']
        instance.trust_model = model_data['trust_model']
        instance.resilience_scaler = model_data['resilience_scaler']
        instance.resilience_features = model_data['resilience_features']
        instance.resilience_model = model_data['resilience_model']
        instance.adapt_scaler = model_data['adapt_scaler']
        instance.adapt_features = model_data['adapt_features']
        instance.adapt_mortgage_cols = model_data['adapt_mortgage_cols']
        instance.adapt_model = model_data['adapt_model']
        instance.lang_vectorizer = model_data['lang_vectorizer']
        instance.lang_model = model_data['lang_model']
        instance.meta_model = model_data['meta_model']
        instance.ae_scaler = model_data['ae_scaler']
        instance.autoencoder = model_data['autoencoder']
        instance.encoder = model_data['encoder']
        instance.decoder = model_data['decoder']
        instance.sentiment_analyzer = model_data['sentiment_analyzer']

        # schema (fallback if older pickle doesn't have it)
        instance.df_columns = model_data.get('df_columns', list(set(
            ['connections','defaulter_neighbors','verified_neighbors',
             'monthly_credit_bills','bnpl_utilization_rate','mortgage_months_left',
             'upi_balances','emi_status_log','income-expense ratio','owns_home',
             'monthly_rent','recovery_days','mortgage_status','user_posts','default_label']
            + instance.resilience_features + instance.adapt_features
        )))

        # no training df after load
        instance.df = None

        print(f"Model loaded from {filepath}")
        return instance


# Train/evaluate/save when executed directly
if __name__ == '__main__':
    scorer = AltiCredScorer()
    scorer.evaluate_model()
    scorer.save_model()
    print("\n--- Example prediction ---")
    test_user = {
        'connections': 'user_1,user_2,user_3',
        'defaulter_neighbors': 1,
        'verified_neighbors': 5,
        'monthly_credit_bills': 15000.0,
        'bnpl_utilization_rate': 0.3,
        'mortgage_months_left': 120.0,
        'upi_balances': '[1500, 2000, 1800]',
        'emi_status_log': '[1, 1, 1, 0, 1]',
        'income-expense ratio': 1.2,
        'owns_home': 1,
        'monthly_rent': 0.0,
        'recovery_days': 3,
        'mortgage_status': 'ongoing',
        'user_posts': 'I had a great week at work and got a promotion!'
    }
    score = scorer.predict_alticred_score(test_user)
    print(f"Test AltiCred Score: {score:.4f}")
