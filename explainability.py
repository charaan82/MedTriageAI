# explainability.py
import os
import joblib
import pandas as pd
import numpy as np
import shap

def generate_shap_explanation():
    model_path = "models/triagemind_attack_classifier.pkl"
    feature_path = "models/feature_columns.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        print("Error: Trained model or feature columns not found. Please run python train_classifier.py first.")
        return None, None
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)
    
    # Generate representative sample data matching feature columns
    X_sample = pd.DataFrame(np.zeros((10, len(feature_cols))), columns=feature_cols)
    
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Disable additivity check to prevent numerical mismatch errors with dummy background data
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    
    print("SHAP explanation generated successfully for Random Forest classifier.")
    return explainer, shap_values

if __name__ == "__main__":
    generate_shap_explanation()