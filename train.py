# train_and_save_model.py

import os
import sys

# Add the current directory to Python path to import the model
sys.path.insert(0, '.')

from models.alticred_salaried import AltiCredScorer

def main():
    """
    Train the AltiCred model and save it as a pickle file.
    """
    print("=" * 60)
    print("AltiCred Model Training and Saving Script")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        # Initialize and train the model
        print("\nInitializing AltiCred Scorer...")
        scorer = AltiCredScorer('data/salaried_dataset.csv')
        
        # Evaluate the model
        scorer.evaluate_model()
        
        # Save the model
        model_path = 'models/alticred_salaried_model.pkl'
        scorer.save_model(model_path)
        
        print(f"\n✅ Model successfully saved to: {model_path}")
        print(f"📊 Model is ready for use in the Flask application!")
        
        # Test the saved model by loading it
        print("\n--- Testing Model Loading ---")
        loaded_scorer = AltiCredScorer.load_model(model_path)
        
        # Test prediction with sample data
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
        
        score = loaded_scorer.predict_alticred_score(test_user)
        print(f"✅ Test prediction successful! Score: {score:.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 Model training and saving completed successfully!")
        print("🚀 You can now run the Flask application with: python app.py")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the dataset file exists at 'data/salaried_dataset.csv'")
        
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        print("Please check your data and try again.")

if __name__ == '__main__':
    main()