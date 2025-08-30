from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables for models and scalers
models = {}
meta_model = None
scaler_rep = None
rep_features = []
Z_cols = []
encoder = None
pca = None
AE_AVAILABLE = False

def load_models():
    """Load all trained models and preprocessing objects"""
    global models, meta_model, scaler_rep, rep_features, Z_cols, encoder, pca, AE_AVAILABLE
    
    models_dir = "models"
    
    try:
        # Load meta model
        with open(os.path.join(models_dir, "meta_model.pkl"), "rb") as f:
            meta_data = pickle.load(f)
            meta_model = meta_data["model"]
        
        # Load component models
        component_names = ["digital_trust", "resilience", "adaptability", "language_sentiment"]
        for name in component_names:
            with open(os.path.join(models_dir, f"{name}_model.pkl"), "rb") as f:
                models[name] = pickle.load(f)
        
        # Load representation features and scaler
        with open(os.path.join(models_dir, "rep_features_and_embeds.pkl"), "rb") as f:
            rep_data = pickle.load(f)
            rep_features = rep_data["rep_features"]
            Z_cols = rep_data["Z_cols"]
            scaler_rep = rep_data["scaler_rep"]
        
        print("All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def get_embeddings(user_data):
    """Generate embeddings for user data"""
    global encoder, pca, AE_AVAILABLE
    
    # Ensure all rep_features columns exist
    user_df = pd.DataFrame([user_data])
    for c in rep_features:
        if c not in user_df.columns:
            user_df[c] = 0.0
    
    # Scale the data
    user_scaled = scaler_rep.transform(user_df[rep_features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    
    # Try to use autoencoder if available, fallback to PCA
    try:
        if AE_AVAILABLE and encoder is not None:
            user_Z = encoder.predict(user_scaled)
        else:
            # Use PCA fallback
            if pca is None:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=len(Z_cols))
                # We need to fit on some data - using zeros as fallback
                dummy_data = np.zeros((10, len(rep_features)))
                pca.fit(dummy_data)
            user_Z = pca.transform(user_scaled)
    except Exception as e:
        print(f"Error in embeddings: {e}")
        # Fallback to zeros
        user_Z = np.zeros((1, len(Z_cols)))
    
    return user_Z

def predict_score(user_input):
    """Predict AltiCred score for a user"""
    try:
        # Get embeddings
        user_Z = get_embeddings(user_input)
        user_embed = pd.DataFrame(user_Z, columns=Z_cols)
        
        # Create full user dataframe
        user_df = pd.DataFrame([user_input])
        user_full = pd.concat([user_df, user_embed], axis=1)
        
        # Predict components
        base_preds = {}
        component_scores = {}
        
        for name, obj in models.items():
            feats = obj["features"]
            # Ensure missing features are present
            for f in feats:
                if f not in user_full.columns:
                    user_full[f] = 0.0
            
            pred = obj["model"].predict(user_full[feats])[0]
            clipped_pred = float(np.clip(pred, 0, 1))
            base_preds[f"{name}_pred"] = clipped_pred
            component_scores[name] = clipped_pred
        
        # Meta prediction
        meta_input = pd.DataFrame([base_preds])
        # Ensure all required columns are present
        required_cols = ["digital_trust_pred", "resilience_pred", "adaptability_pred", "language_sentiment_pred"]
        for col in required_cols:
            if col not in meta_input.columns:
                meta_input[col] = 0.0
        
        final_score = float(np.clip(meta_model.predict(meta_input[required_cols])[0], 0, 1))
        
        return {
            "final_score": final_score,
            "component_scores": component_scores,
            "success": True
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            "error": str(e),
            "success": False
        }

def get_interpretation(score):
    """Get human-readable interpretation of the score"""
    if score >= 0.8:
        return "Excellent creditworthiness"
    elif score >= 0.6:
        return "Good creditworthiness"
    elif score >= 0.4:
        return "Fair creditworthiness — manual review recommended"
    else:
        return "Higher risk — careful evaluation required"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        user_input = {}
        
        # Primary questions mapping
        form_to_feature = {
            'land_size': 'land_size_acres',
            'in_cooperative': 'in_cooperative',
            'linked_to_exporter': 'linked_to_exporter',
            'agritech_usage': 'agritech_tool_usage',
            'new_crop_adoption': 'new_crop_adoption_flag',
            'pm_kisan_installments': 'pm_kisan_installments_received',
            'crop_diversity': 'crop_diversity',
            'digital_engagement': 'digital_network_engagement_value',
            'market_access': 'market_access_value',
            'budgeting_habit': 'budgeting_habit_value'
        }
        
        # Process form inputs
        for form_key, feature_name in form_to_feature.items():
            value = request.form.get(form_key, '0').strip()
            try:
                user_input[feature_name] = float(value) if value else 0.0
            except ValueError:
                user_input[feature_name] = 0.0
        
        # Add derived features with defaults
        user_input['irrigation_proxy'] = 0.0
        user_input['reliable_contacts_count'] = user_input.get('digital_network_engagement_value', 0) * 10
        user_input['social_connections_count'] = user_input.get('digital_network_engagement_value', 0) * 15
        user_input['proximity_to_defaulters_score'] = 0.1
        user_input['support_request_frequency'] = 0.1
        user_input['time_to_resume_upi_after_shock'] = 2.0
        user_input['emi_status_last_12_months'] = 0.1
        user_input['overdraft_usage_frequency'] = 0.1
        user_input['loan_repayments_done'] = 0.8
        user_input['yield_recovered_units'] = 0.7
        user_input['income_volatility_value'] = 0.3
        
        # Make prediction
        result = predict_score(user_input)
        
        if result['success']:
            interpretation = get_interpretation(result['final_score'])
            return jsonify({
                'success': True,
                'final_score': round(result['final_score'], 4),
                'component_scores': {k: round(v, 4) for k, v in result['component_scores'].items()},
                'interpretation': interpretation
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0 and meta_model is not None
    })

if __name__ == '__main__':
    # Initialize models
    if not load_models():
        print("Warning: Models not loaded. Please ensure models/ directory exists with trained models.")
        print("Run train_alticred.py first to generate the models.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)