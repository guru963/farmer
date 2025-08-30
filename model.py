# train_alticred.py
# Robust training pipeline for AltiCred-style scoring (rule-based targets + learned models)
# Paste into a file and run: python train_alticred.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.utils import resample
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
import pickle

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "farmer.csv"  # change if needed
RANDOM_STATE = 42
USE_TF_AUTOENCODER = True   # try TF; will fallback to PCA
LATENT_DIM = 8              # embedding size (PCA or AE)
OSAVE = "models"
os.makedirs(OSAVE, exist_ok=True)

# -----------------------
# SAFE HELPERS
# -----------------------
def col_or_zero(df, name):
    return df[name] if name in df.columns else pd.Series(0.0, index=df.index)

def safe_sum(df, cols):
    if not cols:
        return pd.Series(0.0, index=df.index)
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].sum(axis=1)

def sanitize_series(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def print_scores(tag, y_true, y_pred):
    print(f"{tag} -> R²: {r2_score(y_true, y_pred):.4f} | RMSE: {rmse(y_true, y_pred):.4f} | MAE: {mean_absolute_error(y_true, y_pred):.4f}")

# -----------------------
# 1) Load dataset
# -----------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {DATA_PATH}")

# -----------------------
# 2) Identify candidate simple features (user-facing)
# These are expected or derived from your cleaned numeric file.
# -----------------------
# Numeric/simple features we prefer (most are produced by your cleaning pipeline)
candidate_feats = {
    "land_size": ["land_size_acres", "land_size"],
    "in_cooperative": ["in_cooperative"],
    "linked_to_exporter": ["linked_to_exporter"],
    "agritech_tool_usage": ["agritech_tool_usage"],
    "new_crop_adoption_flag": ["new_crop_adoption_flag"],
    "pm_kisan": ["pm_kisan_installments_received", "pm_kisan_received"],  # count or binary
    "crop_flags": [c for c in df.columns if c.startswith("crop_")],
    "utility_flags": [c for c in df.columns if c.startswith("utility_")],
    # behavioral/engagement proxies (may be present)
    "digital_network_engagement_value": ["digital_network_engagement_value"],
    "market_access_value": ["market_access_value"],
    "social_connections_count": ["social_connections_count"],
    "reliable_contacts_count": ["reliable_contacts_count"],
    "proximity_to_defaulters_score": ["proximity_to_defaulters_score"],
    "support_request_frequency": ["support_request_frequency"],
    # resilience/adaptability inputs
    "time_to_resume_upi_after_shock": ["time_to_resume_upi_after_shock"],
    "emi_status_last_12_months": ["emi_status_last_12_months"],
    "overdraft_usage_frequency": ["overdraft_usage_frequency"],
    "loan_repayments_done": ["loan_repayments_done", "loan_repayment_ratio"],
    "yield_recovered_units": ["yield_recovered_units", "yield_recovery_ratio"],
    "income_volatility_value": ["income_volatility_value"],
    "budgeting_habit_value": ["budgeting_habit_value"],
}

# create easy access variables (prefer first available column name)
def pick_col(df, candidates):
    for c in candidates:
        if isinstance(c, str) and c in df.columns:
            return df[c]
    return pd.Series(0.0, index=df.index)

# Build simple series using available columns (fall back to zeros)
series = {}
for key, cand in candidate_feats.items():
    if isinstance(cand, list) and len(cand) > 0 and isinstance(cand[0], str) and cand[0].startswith("crop_"):
        # already precomputed crop flags list
        crop_cols = [c for c in df.columns if c.startswith("crop_")]
        series["crop_diversity"] = safe_sum(df, crop_cols)
    else:
        series[key] = pick_col(df, cand) if not isinstance(cand, list) or not cand or not cand[0].startswith("crop_") else pd.Series(0.0, index=df.index)

# Some boolean conversions / proxies
df["pm_kisan_binary"] = (sanitize_series(col_or_zero(df, "pm_kisan_installments_received")) > 0).astype(float)
df["irrigation_proxy"] = (safe_sum(df, [c for c in df.columns if "pump" in c.lower() or "bore" in c.lower() or "irrig" in c.lower()]) > 0).astype(float)

# ensure key series in df for formulas (use either existing column or our proxies)
df["land_size_acres"] = sanitize_series(col_or_zero(df, "land_size_acres"))
df["in_cooperative"] = sanitize_series(col_or_zero(df, "in_cooperative"))
df["linked_to_exporter"] = sanitize_series(col_or_zero(df, "linked_to_exporter"))
df["agritech_tool_usage"] = sanitize_series(col_or_zero(df, "agritech_tool_usage"))
df["new_crop_adoption_flag"] = sanitize_series(col_or_zero(df, "new_crop_adoption_flag"))
df["crop_diversity"] = series.get("crop_diversity", pd.Series(0.0, index=df.index))
df["digital_network_engagement_value"] = sanitize_series(col_or_zero(df, "digital_network_engagement_value"))
df["market_access_value"] = sanitize_series(col_or_zero(df, "market_access_value"))
df["social_connections_count"] = sanitize_series(col_or_zero(df, "social_connections_count"))
df["reliable_contacts_count"] = sanitize_series(col_or_zero(df, "reliable_contacts_count"))
df["proximity_to_defaulters_score"] = sanitize_series(col_or_zero(df, "proximity_to_defaulters_score"))
df["support_request_frequency"] = sanitize_series(col_or_zero(df, "support_request_frequency"))
df["time_to_resume_upi_after_shock"] = sanitize_series(col_or_zero(df, "time_to_resume_upi_after_shock"))
df["emi_status_last_12_months"] = sanitize_series(col_or_zero(df, "emi_status_last_12_months"))
df["overdraft_usage_frequency"] = sanitize_series(col_or_zero(df, "overdraft_usage_frequency"))
df["loan_repayments_done"] = sanitize_series(col_or_zero(df, "loan_repayments_done"))
df["yield_recovered_units"] = sanitize_series(col_or_zero(df, "yield_recovered_units"))
df["income_volatility_value"] = sanitize_series(col_or_zero(df, "income_volatility_value"))
df["budgeting_habit_value"] = sanitize_series(col_or_zero(df, "budgeting_habit_value"))
df["pm_kisan_installments_received"] = sanitize_series(col_or_zero(df, "pm_kisan_installments_received"))

# -----------------------
# 3) Compute component raw scores (transparent weights)
#    We use simple linear combinations and then scale 0-1
# -----------------------
def scale01(s):
    s = pd.to_numeric(s, errors="coerce")
    if (s.max() - s.min()) == 0:
        return (s * 0.0).fillna(0.0)
    return (s - s.min()) / (s.max() - s.min())

df["digital_trust_raw"] = (
    0.30 * df["digital_network_engagement_value"]
    + 0.20 * df["reliable_contacts_count"]
    + 0.10 * df["social_connections_count"]
    - 0.25 * df["proximity_to_defaulters_score"]
    - 0.10 * df["support_request_frequency"]
    + 0.15 * df["market_access_value"]
    + 0.20 * df["in_cooperative"]
    + 0.10 * df["linked_to_exporter"]
)

df["resilience_raw"] = (
    0.25 * df["land_size_acres"]
    + 0.25 * df["irrigation_proxy"]
    - 0.20 * df["time_to_resume_upi_after_shock"]
    - 0.15 * df["emi_status_last_12_months"]
    - 0.10 * df["overdraft_usage_frequency"]
    + 0.20 * df["loan_repayments_done"]
    + 0.20 * df["yield_recovered_units"]
    + 0.10 * df["pm_kisan_installments_received"]
)

df["adaptability_raw"] = (
    -0.30 * df["income_volatility_value"]
    + 0.30 * df["yield_recovered_units"]
    + 0.25 * df["budgeting_habit_value"]
    + 0.30 * df["agritech_tool_usage"]
    + 0.15 * df["loan_repayments_done"]
    + 0.20 * df["new_crop_adoption_flag"]
    + 0.10 * df["crop_diversity"]
)

df["language_raw"] = (
    0.30 * df["budgeting_habit_value"]
    + 0.20 * df["digital_network_engagement_value"]
    + 0.15 * df["market_access_value"]
    + 0.25 * df["agritech_tool_usage"]
    + 0.10 * df["in_cooperative"]
)

# scale to 0-1
df["digital_trust"] = scale01(df["digital_trust_raw"])
df["resilience"] = scale01(df["resilience_raw"])
df["adaptability"] = scale01(df["adaptability_raw"])
df["language_sentiment"] = scale01(df["language_raw"])

# final blended target (meta-target)
df["final_behavioral_score"] = (
    0.25 * df["digital_trust"] +
    0.30 * df["resilience"] +
    0.25 * df["adaptability"] +
    0.20 * df["language_sentiment"]
)

# -----------------------
# 4) Representation learning (AutoEncoder or PCA)
# -----------------------
rep_features = sorted(list(set([
    "digital_network_engagement_value", "reliable_contacts_count", "social_connections_count",
    "proximity_to_defaulters_score", "support_request_frequency", "market_access_value",
    "land_size_acres", "irrigation_proxy", "time_to_resume_upi_after_shock",
    "emi_status_last_12_months", "overdraft_usage_frequency", "loan_repayments_done",
    "yield_recovered_units", "pm_kisan_installments_received", "income_volatility_value",
    "budgeting_habit_value", "agritech_tool_usage", "new_crop_adoption_flag", "crop_diversity"
])))

# select only those present
rep_features = [c for c in rep_features if c in df.columns]

X_rep = df[rep_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
scaler_rep = StandardScaler()
X_scaled = scaler_rep.fit_transform(X_rep)

AE_AVAILABLE = False
Z_all = None
try:
    if USE_TF_AUTOENCODER:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        input_dim = X_scaled.shape[1]
        inp = keras.Input(shape=(input_dim,))
        h = layers.Dense(max(32, input_dim), activation="relu")(inp)
        z = layers.Dense(LATENT_DIM, activation="linear", name="latent")(h)
        h2 = layers.Dense(max(32, input_dim), activation="relu")(z)
        out = layers.Dense(input_dim, activation="linear")(h2)
        ae = keras.Model(inp, out)
        ae.compile(optimizer="adam", loss="mse")
        ae.fit(X_scaled, X_scaled, epochs=30, batch_size=32, verbose=0)
        encoder = keras.Model(inp, z)
        Z_all = encoder.predict(X_scaled, verbose=0)
        AE_AVAILABLE = True
except Exception:
    AE_AVAILABLE = False

if not AE_AVAILABLE:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(LATENT_DIM, X_scaled.shape[1]))
    Z_all = pca.fit_transform(X_scaled)

Z_cols = [f"embed_{i+1}" for i in range(Z_all.shape[1])]
df_embed = pd.DataFrame(Z_all, columns=Z_cols, index=df.index)
df_full = pd.concat([df, df_embed], axis=1)

# -----------------------
# 5) Train/test split (no label required; we use computed targets)
#    If default_flag exists we can stratify; otherwise regular split
# -----------------------
if "default_flag" in df_full.columns:
    strat = df_full["default_flag"].astype(int)
else:
    strat = None

train_idx, test_idx = train_test_split(df_full.index, test_size=0.20, random_state=RANDOM_STATE, stratify=strat)
train_df = df_full.loc[train_idx].copy()
test_df = df_full.loc[test_idx].copy()

# -----------------------
# 6) Feature maps (use numeric-only columns present)
# -----------------------
features_map = {
    "digital_trust": [c for c in [
        "digital_network_engagement_value", "reliable_contacts_count", "social_connections_count",
        "proximity_to_defaulters_score", "support_request_frequency", "market_access_value", "in_cooperative", "linked_to_exporter"
    ] if c in df_full.columns],
    "resilience": [c for c in [
        "land_size_acres", "irrigation_proxy", "time_to_resume_upi_after_shock",
        "emi_status_last_12_months", "overdraft_usage_frequency", "loan_repayments_done", "yield_recovered_units", "pm_kisan_installments_received"
    ] if c in df_full.columns],
    "adaptability": [c for c in [
        "income_volatility_value", "yield_recovered_units", "budgeting_habit_value",
        "agritech_tool_usage", "loan_repayments_done", "new_crop_adoption_flag", "crop_diversity"
    ] if c in df_full.columns],
    "language_sentiment": [c for c in [
        "budgeting_habit_value", "digital_network_engagement_value", "market_access_value", "agritech_tool_usage", "in_cooperative"
    ] if c in df_full.columns]
}

targets = {
    "digital_trust": "digital_trust",
    "resilience": "resilience",
    "adaptability": "adaptability",
    "language_sentiment": "language_sentiment"
}

# -----------------------
# 7) Training helper functions
# -----------------------
def train_xgb_with_grid(X, y):
    y = sanitize_series(y).values
    model = XGBRegressor(random_state=RANDOM_STATE, verbosity=0, tree_method="hist", n_jobs=-1, objective="reg:squarederror")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    }
    grid = GridSearchCV(model, param_grid, cv=3, scoring=make_scorer(r2_score), n_jobs=-1, error_score='raise')
    grid.fit(X, y)
    return grid.best_estimator_

def train_lasso(X, y, alpha=0.01):
    y = sanitize_series(y).values
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X, y)
    return model

# -----------------------
# 8) Train component models
# -----------------------
models = {}
component_metrics = {}

for name, feats in features_map.items():
    # include embeddings
    if not feats:
        print(f"WARNING: No features found for component {name}; skipping.")
        continue
    X_train = train_df[feats + Z_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = sanitize_series(train_df[targets[name]])
    X_test = test_df[feats + Z_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = sanitize_series(test_df[targets[name]])

    if name in ["digital_trust", "resilience", "adaptability"]:
        model = train_xgb_with_grid(X_train, y_train)
    else:
        model = train_lasso(X_train, y_train, alpha=0.01)

    models[name] = {"model": model, "features": feats + Z_cols}

    # predict and clip to [0,1]
    yhat_train = np.clip(model.predict(X_train), 0, 1)
    yhat_test  = np.clip(model.predict(X_test), 0, 1)

    component_metrics[name] = {
        "train_r2": r2_score(y_train, yhat_train),
        "train_rmse": rmse(y_train, yhat_train),
        "train_mae": mean_absolute_error(y_train, yhat_train),
        "test_r2": r2_score(y_test, yhat_test),
        "test_rmse": rmse(y_test, yhat_test),
        "test_mae": mean_absolute_error(y_test, yhat_test),
    }

    print(f"\n=== Component: {name.upper()} ===")
    print_scores("Train", y_train, yhat_train)
    print_scores("Test", y_test, yhat_test)

# -----------------------
# 9) Meta-model (stacking)
# -----------------------
# Build meta features from each component's predictions
meta_train = {}
meta_test = {}
for name, obj in models.items():
    model = obj["model"]
    feats = obj["features"]
    meta_train[f"{name}_pred"] = model.predict(train_df[feats])
    meta_test[f"{name}_pred"]  = model.predict(test_df[feats])

meta_X_train = pd.DataFrame(meta_train).clip(0,1)
meta_X_test  = pd.DataFrame(meta_test).clip(0,1)
meta_y_train = sanitize_series(train_df["final_behavioral_score"])
meta_y_test  = sanitize_series(test_df["final_behavioral_score"])

meta_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
meta_model.fit(meta_X_train, meta_y_train)
meta_pred_tr = np.clip(meta_model.predict(meta_X_train), 0, 1)
meta_pred_te = np.clip(meta_model.predict(meta_X_test), 0, 1)

print("\n=== META MODEL ===")
print_scores("Meta Train", meta_y_train, meta_pred_tr)
print_scores("Meta Test",  meta_y_test, meta_pred_te)

# print meta weights
print("\nMeta-model coefficients:")
for fname, coef in zip(meta_X_train.columns, meta_model.coef_):
    print(f"  {fname}: {coef:.4f}")

# -----------------------
# 10) Persist models & artifacts
# -----------------------
with open(os.path.join(OSAVE, "meta_model.pkl"), "wb") as f:
    pickle.dump({"model": meta_model, "features": list(meta_X_train.columns)}, f)
for name, obj in models.items():
    with open(os.path.join(OSAVE, f"{name}_model.pkl"), "wb") as f:
        pickle.dump(obj, f)
with open(os.path.join(OSAVE, "rep_features_and_embeds.pkl"), "wb") as f:
    pickle.dump({"rep_features": rep_features, "Z_cols": Z_cols, "scaler_rep": scaler_rep}, f)

print(f"\nSaved models to {OSAVE}/")

# -----------------------
# 11) Interactive prediction for a new farmer (ask only primary inputs)
# -----------------------
print("\n=== Enter user input for primary fields (press enter to default 0) ===")
# collect only farmer-facing primary questions (simple)
primary_questions = {
    "land_size_acres": "Land size (acres)",
    "in_cooperative": "In cooperative? (1=yes,0=no)",
    "linked_to_exporter": "Linked to exporter? (1=yes,0=no)",
    "agritech_tool_usage": "Uses agritech tools? (1=yes,0=no)",
    "new_crop_adoption_flag": "Adopted new crop recently? (1=yes,0=no)",
    "pm_kisan_installments_received": "PM-Kisan installments received (count)",
    "crop_diversity": "Number of distinct crops grown",
    "digital_network_engagement_value": "Digital engagement score (0-1)",
    "market_access_value": "Market access score (0-1)",
    "budgeting_habit_value": "Budgeting habit score (0-1)"
}

user_input = {}
for col, prompt in primary_questions.items():
    val = input(f"{prompt}: ").strip()
    if val == "":
        user_input[col] = 0.0
    else:
        try:
            user_input[col] = float(val)
        except:
            user_input[col] = 0.0

# build user DataFrame
user_df = pd.DataFrame([user_input])

# ensure all rep_features columns exist in user_df (fill zeros if missing)
for c in rep_features:
    if c not in user_df.columns:
        user_df[c] = 0.0

# scale using scaler_rep and get embedding
user_scaled = scaler_rep.transform(user_df[rep_features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
if AE_AVAILABLE:
    # reuse the encoder from earlier if present
    try:
        user_Z = encoder.predict(user_scaled)
    except:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=len(Z_cols))
        pca.fit(X_scaled)
        user_Z = pca.transform(user_scaled)
else:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=len(Z_cols))
    pca.fit(X_scaled)
    user_Z = pca.transform(user_scaled)

user_embed = pd.DataFrame(user_Z, columns=Z_cols)
user_full = pd.concat([user_df, user_embed], axis=1)

# Predict components
base_preds = {}
for name, obj in models.items():
    feats = obj["features"]
    # ensure missing features are present
    for f in feats:
        if f not in user_full.columns:
            user_full[f] = 0.0
    pred = obj["model"].predict(user_full[feats])[0]
    base_preds[f"{name}_pred"] = float(np.clip(pred, 0, 1))

# Meta prediction
meta_input = pd.DataFrame([base_preds])[list(meta_X_train.columns)]
final_score = float(np.clip(meta_model.predict(meta_input)[0], 0, 1))

print("\n=== Prediction Results ===")
for k, v in base_preds.items():
    print(f" {k}: {v:.4f}")
print(f" Final AltiCred Score: {final_score:.4f}")

# Interpretation
if final_score >= 0.8:
    print("Interpretation: Excellent creditworthiness")
elif final_score >= 0.6:
    print("Interpretation: Good creditworthiness")
elif final_score >= 0.4:
    print("Interpretation: Fair creditworthiness — manual review recommended")
else:
    print("Interpretation: Higher risk — careful evaluation required")
