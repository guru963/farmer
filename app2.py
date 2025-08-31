# app.py

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
import json

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load the Trained Model ---
def load_alticred_model():
    """Load the trained AltiCred model from pickle file."""
    model_path = 'models/alticred_salaried_model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train and save the model first.")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

# --- Helper Classes for Model Loading ---
class AltiCredPredictor:
    """
    A lightweight class to handle predictions using the loaded model data.
    """
    def __init__(self, model_data):
        # Load all model components
        self.trust_scaler = model_data['trust_scaler']
        self.trust_encoder = model_data['trust_encoder']
        self.trust_model = model_data['trust_model']
        self.resilience_scaler = model_data['resilience_scaler']
        self.resilience_features = model_data['resilience_features']
        self.resilience_model = model_data['resilience_model']
        self.adapt_scaler = model_data['adapt_scaler']
        self.adapt_features = model_data['adapt_features']
        self.adapt_mortgage_cols = model_data['adapt_mortgage_cols']
        self.adapt_model = model_data['adapt_model']
        self.lang_vectorizer = model_data['lang_vectorizer']
        self.lang_model = model_data['lang_model']
        self.meta_model = model_data['meta_model']
        self.sentiment_analyzer = model_data['sentiment_analyzer']
        self.df_columns = model_data['df_columns']

    def safe_json_list_parser(self, x):
        """
        Safely parses a string that might be a number, a comma-separated list,
        or a JSON list. Always returns a list.
        """
        try:
            s = str(x).strip()
            if not s.startswith('['):
                s = f"[{s}]"
            return json.loads(s.replace("'", "\""))
        except:
            return []

    def _predict_digital_trust_score(self, data):
        df = data.copy()
        # Ensure all required columns exist
        for col in self.df_columns:
            if col not in df.columns:
                df[col] = 0
        
        df['num_connections'] = df['connections'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        X = df[['defaulter_neighbors', 'verified_neighbors', 'num_connections']]
        X_scaled = self.trust_scaler.transform(X)
        X_encoded = self.trust_encoder.predict(X_scaled)
        return self.trust_model.predict_proba(X_encoded)[:, 1]

    def _predict_resilience_score(self, data):
        df = data.copy()
        # Ensure all required columns exist
        for col in self.df_columns:
            if col not in df.columns:
                df[col] = 0
                
        df['avg_upi_balance'] = df['upi_balances'].apply(lambda x: np.mean(self.safe_json_list_parser(x)) if self.safe_json_list_parser(x) else 0)
        df['missed_emi_count'] = df['emi_status_log'].apply(lambda x: self.safe_json_list_parser(x).count(0))
        X = df[self.resilience_features]
        X_scaled = self.resilience_scaler.transform(X)
        return self.resilience_model.predict_proba(X_scaled)[:, 1]

    def _predict_adaptability_score(self, data):
        df = data.copy()
        # Ensure all required columns exist
        for col in self.df_columns:
            if col not in df.columns:
                df[col] = 0
                
        df['missed_emi_count'] = df['emi_status_log'].apply(lambda x: self.safe_json_list_parser(x).count(0))
        mortgage_dummies = pd.get_dummies(df['mortgage_status'], prefix='mortgage')
        df = pd.concat([df, mortgage_dummies], axis=1)
        for col in self.adapt_mortgage_cols:
            if col not in df.columns: 
                df[col] = 0
        X = df[self.adapt_features]
        X_scaled = self.adapt_scaler.transform(X)
        return self.adapt_model.predict_proba(X_scaled)[:, 1]

    def _predict_language_sentiment_score(self, data):
        import re
        from scipy.sparse import hstack
        
        df = data.copy()
        # Ensure all required columns exist
        for col in self.df_columns:
            if col not in df.columns:
                df[col] = 0
                
        df['cleaned_posts'] = df['user_posts'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
        
        get_score = lambda text: self.sentiment_analyzer.polarity_scores(text)['compound']
        df['generated_sentiment_score'] = df['user_posts'].apply(get_score)
        
        X_text = self.lang_vectorizer.transform(df['cleaned_posts'])
        X_sentiment = df['generated_sentiment_score'].values.reshape(-1, 1)
        
        X_combined = hstack([X_text, X_sentiment])
        return self.lang_model.predict_proba(X_combined)[:, 1]

    def predict_alticred_score(self, user_data):
        """
        Predicts the AltiCred score for a given user.
        Returns only the final score as a float.
        """
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

# --- Load Model on Startup ---
print("Loading AltiCred model... Please wait.")
try:
    model_data = load_alticred_model()
    salaried_scorer = AltiCredPredictor(model_data)
    print("AltiCred model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have trained and saved the model first by running the training script.")
    exit(1)

# --- Helper function for safe numeric conversion ---
def safe_float(value):
    """
    Safely converts a value to float.
    Returns 0.0 if value is None, empty, or cannot be converted.
    """
    try:
        # Handle booleans stored as strings
        if str(value).lower() in ['true', 'yes']:
            return 1.0
        elif str(value).lower() in ['false', 'no']:
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0  # Default value if invalid

# --- Define a Route for the Frontend ---
@app.route('/')
def home():
    """Renders the main user interface page."""
    return render_template('index2.html')

# --- Define a Route for Scoring ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives user data, selects a model, and returns a score."""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        user_input = data.get('user_data')

        if not model_type or not user_input:
            return jsonify({'error': 'Missing model_type or user_data'}), 400

        # Fields that must be numeric
        numeric_fields = [
            'defaulter_neighbors', 'verified_neighbors', 'monthly_credit_bills',
            'bnpl_utilization_rate', 'mortgage_months_left', 'income-expense ratio',
            'owns_home', 'monthly_rent', 'recovery_days'
        ]

        # Safely convert all numeric fields
        for field in numeric_fields:
            if field in user_input:
                user_input[field] = safe_float(user_input[field])

        # --- Select the Model and Predict ---
        if model_type == 'salaried':
            score = salaried_scorer.predict_alticred_score(user_input)
        else:
            return jsonify({'error': 'Invalid model type specified. Only "salaried" is currently supported.'}), 400

        return jsonify({'alticred_score': f"{score:.4f}"})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

# --- Health Check Route ---
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': True})

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
