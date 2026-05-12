# ============================================
# BACKEND: Model Loading and Prediction Logic
# ============================================

import joblib
import pandas as pd
import os
from typing import Dict, Any, Tuple

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")

# Global variables to cache model and components
_model = None
_features = None
_metrics = None
_feature_importance = None

# Define the features in the correct order
FEATURES = ['age', 'bmi', 'children', 'female_dm', 'smoker_dm',
            'smoker_bmi_interaction', 'age_squared']

# Model performance metrics (from your results)
MODEL_METRICS = {
    "r2_score": 0.8938,
    "mae": 2521,
    "rmse": 4418,
    "model_type": "Random Forest Regressor (Refined)",
    "train_samples": 1069,
    "test_samples": 268
}

# Feature importance from your refined model
FEATURE_IMPORTANCE = {
    "smoker_bmi_interaction": {"name": "Smoker × BMI Interaction", "importance": 0.429, "percent": "42.9%"},
    "smoker_dm": {"name": "Smoking Status", "importance": 0.354, "percent": "35.4%"},
    "age": {"name": "Age", "importance": 0.069, "percent": "6.9%"},
    "age_squared": {"name": "Age² (Non-linear Age)", "importance": 0.068, "percent": "6.8%"},
    "bmi": {"name": "BMI", "importance": 0.062, "percent": "6.2%"},
    "children": {"name": "Number of Children", "importance": 0.014, "percent": "1.4%"},
    "female_dm": {"name": "Gender (Female)", "importance": 0.004, "percent": "0.4%"}
}

# Business insights
BUSINESS_INSIGHTS = {
    "top_predictor": {
        "title": "Smoking-BMI Interaction",
        "importance": "42.9%",
        "insight": "Smokers with high BMI have exponentially higher costs"
    },
    "smoker_impact": {
        "title": "Smoking Status",
        "importance": "35.4%", 
        "insight": "Smokers pay ~$23,610 more on average"
    },
    "gender_impact": {
        "title": "Gender",
        "importance": "0.4%",
        "insight": "Gender has negligible impact → avoid gender-based pricing"
    },
    "annual_savings": {
        "title": "Annual Business Impact",
        "value": "$1.94M",
        "insight": "Savings per 10,000 policies from refinement"
    }
}

def load_model():
    """Load the Random Forest model from disk (cached)."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            print("✅ Backend: Model loaded successfully!")
        except Exception as e:
            print(f"❌ Backend: Error loading model: {e}")
            from sklearn.ensemble import RandomForestRegressor
            _model = RandomForestRegressor(n_estimators=100, random_state=42)
            print("⚠️ Backend: Using placeholder model")
    return _model

def preprocess_features(age: int, bmi: float, children: int, gender: str, smoker: str) -> pd.DataFrame:
    """Convert raw inputs to model-ready features."""
    female_dm = 1 if gender == "Female" else 0
    smoker_dm = 1 if smoker == "Yes" else 0
    
    # Feature engineering
    smoker_bmi_interaction = smoker_dm * bmi
    age_squared = age ** 2
    
    # Create feature array in correct order
    features = pd.DataFrame([[
        age, bmi, children, female_dm, smoker_dm,
        smoker_bmi_interaction, age_squared
    ]], columns=FEATURES)
    
    return features

def predict(age: int, bmi: float, children: int, gender: str, smoker: str) -> float:
    """Main prediction function - called by frontend."""
    model = load_model()
    features = preprocess_features(age, bmi, children, gender, smoker)
    prediction = model.predict(features)[0]
    return round(prediction, 2)

def get_model_metrics() -> Dict[str, Any]:
    """Return model performance metrics."""
    return MODEL_METRICS

def get_feature_importance() -> Dict[str, Dict[str, Any]]:
    """Return feature importance data."""
    return FEATURE_IMPORTANCE

def get_business_insights() -> Dict[str, Any]:
    """Return business insights."""
    return BUSINESS_INSIGHTS

def get_risk_level(prediction: float) -> Tuple[str, str]:
    """Determine risk level based on predicted premium."""
    if prediction > 30000:
        return "High", "#ff6b6b"
    elif prediction > 15000:
        return "Medium", "#ffd93d"
    else:
        return "Standard", "#00d4ff"

# Pre-load model when backend module loads
load_model()