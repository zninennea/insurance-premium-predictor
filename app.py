# ============================================
# Insurance Premium Predictor - Gradio App
# WITH MODEL INFORMATION DISPLAY
# Deployable to Hugging Face Spaces
# ============================================

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")

# Load the trained Random Forest model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("⚠️ Using placeholder model")

# Define the features in the correct order
FEATURES = ['age', 'bmi', 'children', 'female_dm', 'smoker_dm',
            'smoker_bmi_interaction', 'age_squared']

# Model performance metrics (from your results)
MODEL_METRICS = {
    "R² Score": "0.8938 (89.4% variance explained)",
    "Mean Absolute Error (MAE)": "$2,521",
    "Root Mean Square Error (RMSE)": "$4,418",
    "Model Type": "Random Forest Regressor (Refined)",
    "Training Samples": "1,069",
    "Test Samples": "268"
}

# Feature importance from your refined model
FEATURE_IMPORTANCE = {
    "smoker_bmi_interaction (Smoker × BMI)": "42.9%",
    "smoker_dm (Smoking Status)": "35.4%",
    "age": "6.9%",
    "age_squared": "6.8%",
    "bmi": "6.2%",
    "children": "1.4%",
    "female_dm (Gender)": "0.4%"
}

# Business insights
BUSINESS_INSIGHTS = """
📊 **Key Business Insights:**

1. 🔥 **Smoking-BMI Interaction (42.9% importance)** 
   - Smokers with high BMI have exponentially higher costs
   - Target this segment for premium adjustments

2. 🚬 **Smoking Status (35.4% importance)**
   - Smokers pay ~$23,610 more on average
   - Implement smoker verification during application

3. 📈 **Age Effect (13.7% combined)**
   - Costs increase non-linearly with age
   - Premiums should accelerate after age 50

4. ⚧ **Gender (0.4% importance)**
   - Gender has negligible impact on costs
   - Avoid gender-based pricing

5. 💰 **Business Impact**
   - Annual savings: $1.94M per 10,000 policies
   - 7.1% improvement from baseline model
"""

def predict_premium(age, bmi, children, gender, smoker):
    """Predict insurance premium based on user inputs."""
    
    # Convert categorical inputs to numeric
    female_dm = 1 if gender == "Female" else 0
    smoker_dm = 1 if smoker == "Yes" else 0
    
    # Feature engineering (same as during training)
    smoker_bmi_interaction = smoker_dm * bmi
    age_squared = age ** 2
    
    # Create feature array in correct order
    features = pd.DataFrame([[
        age, bmi, children, female_dm, smoker_dm,
        smoker_bmi_interaction, age_squared
    ]], columns=FEATURES)
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Return the prediction with additional info
    return prediction

def create_model_info_html():
    """Create HTML for model information display."""
    
    # Feature importance bars
    feature_html = ""
    for feature, importance in FEATURE_IMPORTANCE.items():
        # Create a simple text-based bar
        percent = float(importance.strip('%'))
        bar_length = int(percent / 2)  # Scale for display
        bar = "█" * bar_length + "░" * (50 - bar_length)
        feature_html += f'<tr><td style="padding: 4px;">{feature}</td><td style="padding: 4px;">{importance}</td><td style="padding: 4px;"><code>{bar}</code></td></tr>'
    
    html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 20px; border-radius: 12px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <h2 style="color: #00d4ff; margin-bottom: 20px;">📊 Model Performance</h2>
        <table style="width: 100%; border-collapse: collapse;">
            {''.join([f'<tr style="border-bottom: 1px solid #333;"><td style="padding: 8px;"><strong>{k}</strong></td><td style="padding: 8px;">{v}</td></tr>' for k, v in MODEL_METRICS.items()])}
        </table>
        
        <h2 style="color: #00d4ff; margin-top: 25px; margin-bottom: 20px;">🔑 Feature Importance</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #0f3460;">
                <th style="padding: 8px; text-align: left;">Feature</th>
                <th style="padding: 8px; text-align: left;">Importance</th>
                <th style="padding: 8px; text-align: left;">Visualization</th>
            </tr>
            {feature_html}
        </table>
        
        <h2 style="color: #00d4ff; margin-top: 25px; margin-bottom: 20px;">💡 Business Insights</h2>
        <div style="background: #0f3460; padding: 15px; border-radius: 8px; line-height: 1.6;">
            <p>✅ <strong>Smoking-BMI Interaction</strong> is the strongest predictor (42.9%)<br>
            <span style="color: #aaa; font-size: 0.9em;">→ Smokers with high BMI have exponentially higher costs</span></p>
            
            <p>✅ <strong>Smoking Status</strong> alone explains 35.4% of cost variance<br>
            <span style="color: #aaa; font-size: 0.9em;">→ Smokers pay ~$23,610 more on average</span></p>
            
            <p>✅ <strong>Gender</strong> has only 0.4% impact<br>
            <span style="color: #aaa; font-size: 0.9em;">→ Avoid gender-based pricing</span></p>
            
            <p>✅ <strong>Annual Business Impact</strong><br>
            <span style="color: #aaa; font-size: 0.9em;">→ $1.94M savings per 10,000 policies from refinement alone</span></p>
        </div>
        
        <div style="text-align: center; margin-top: 20px; font-size: 12px; color: #888;">
            Model: Random Forest (Refined) | R² = 0.8938 | MAE = $2,521
        </div>
    </div>
    """
    return html

def predict_with_display(age, bmi, children, gender, smoker):
    """Prediction function that returns both prediction and model info."""
    prediction = predict_premium(age, bmi, children, gender, smoker)
    
    # Format the prediction result
    result_html = f"""
    <div style="background: linear-gradient(135deg, #00b4d8, #0077b6); color: white; padding: 20px; border-radius: 12px; text-align: center;">
        <h3 style="margin: 0 0 10px 0;">💰 Your Predicted Annual Premium</h3>
        <div style="font-size: 48px; font-weight: bold; margin: 10px 0;">${prediction:,.2f}</div>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">per year</p>
        <hr style="margin: 15px 0; background: rgba(255,255,255,0.3);">
        <p style="font-size: 12px; margin: 0;">Based on your inputs: Age {age}, BMI {bmi}, {gender}, {'Smoker' if smoker == 'Yes' else 'Non-smoker'}</p>
    </div>
    """
    
    return result_html, create_model_info_html()

# Create the Gradio interface with two columns
with gr.Blocks(title="Medical Insurance Premium Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Medical Insurance Premium Predictor
    
    This application predicts annual medical insurance costs based on your personal information.
    The model uses a **Refined Random Forest** algorithm trained on real insurance data.
    
    ---
    """)
    
    with gr.Row():
        # Left Column: Input Form
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Enter Your Details")
            
            age = gr.Slider(18, 64, value=30, step=1, label="👤 Age", info="18-64 years")
            bmi = gr.Slider(15, 53, value=26.0, step=0.1, label="⚖️ BMI", info="Body Mass Index (18.5-24.9 is normal)")
            children = gr.Slider(0, 5, value=0, step=1, label="👶 Children", info="Number of dependents")
            gender = gr.Radio(["Male", "Female"], label="⚧ Gender", value="Male")
            smoker = gr.Radio(["No", "Yes"], label="🚬 Smoker", value="No")
            
            predict_btn = gr.Button("🔮 Predict Premium", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            **📌 Example Inputs** (Click to try):
            """)
            
            gr.Examples(
                examples=[
                    [30, 26.0, 0, "Male", "No"],
                    [50, 30.0, 2, "Female", "Yes"],
                    [25, 22.5, 0, "Female", "No"],
                    [60, 35.0, 1, "Male", "Yes"],
                ],
                inputs=[age, bmi, children, gender, smoker],
                label=None
            )
        
        # Right Column: Results and Model Info
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Your Prediction Result")
            prediction_output = gr.HTML(label="")
            
            gr.Markdown("### 📈 Model Information & Business Insights")
            model_info_output = gr.HTML(create_model_info_html())
    
    # Connect the prediction button
    predict_btn.click(
        fn=predict_with_display,
        inputs=[age, bmi, children, gender, smoker],
        outputs=[prediction_output, model_info_output]
    )
    
    # Also update model info when page loads (not strictly necessary but ensures display)
    demo.load(fn=lambda: (None, create_model_info_html()), outputs=[prediction_output, model_info_output])

# Launch the app
if __name__ == "__main__":
    demo.launch()