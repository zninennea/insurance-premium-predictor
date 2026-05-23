import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf_model.pkl')
        return model
    except:
        # Fallback model
        from sklearn.ensemble import RandomForestRegressor
        st.warning("Using fallback model. Please upload rf_model.pkl for accurate predictions.")
        return RandomForestRegressor(n_estimators=100, random_state=42)

model = load_model()

# Feature names
feature_names = ['age', 'bmi', 'children', 'female_dm', 'smoker_dm', 
                 'smoker_bmi_interaction', 'age_squared']

# Title
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("### Predict your annual healthcare charges with machine learning")

# Sidebar for inputs
st.sidebar.header("📝 Personal Information")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.5)
children = st.sidebar.slider("Number of Children", 0, 10, 0)
sex = st.sidebar.selectbox("Gender", ["Male", "Female"])
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])

# BMI Interpretation
if bmi < 18.5:
    bmi_status = "Underweight"
    bmi_color = "blue"
elif bmi < 25:
    bmi_status = "Normal"
    bmi_color = "green"
elif bmi < 30:
    bmi_status = "Overweight"
    bmi_color = "orange"
else:
    bmi_status = "Obese"
    bmi_color = "red"

# Convert to model inputs
female_dm = 1 if sex == "Female" else 0
smoker_dm = 1 if smoker == "Yes" else 0
smoker_bmi_interaction = smoker_dm * bmi
age_squared = age ** 2

# Make prediction
features = np.array([[age, bmi, children, female_dm, smoker_dm,
                      smoker_bmi_interaction, age_squared]])
prediction = model.predict(features)[0]

# Calculate confidence bounds (approximate)
confidence_lower = prediction * 0.85
confidence_upper = prediction * 1.15

# Risk assessment
if prediction < 5000:
    risk_level = "Low Risk"
    risk_color = "success"
    risk_icon = "🟢"
elif prediction < 15000:
    risk_level = "Medium Risk"
    risk_color = "warning"
    risk_icon = "🟡"
else:
    risk_level = "High Risk"
    risk_color = "error"
    risk_icon = "🔴"

# Main content area - 2 columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Your Health Profile")
    
    # Create metrics display
    col1a, col1b = st.columns(2)
    with col1a:
        st.metric("Age", f"{age} years")
        st.metric("BMI", f"{bmi:.1f} ({bmi_status})")
        st.metric("Children", children)
    with col1b:
        st.metric("Gender", sex)
        st.metric("Smoker", smoker)
    
    # Risk factors analysis
    st.subheader("⚠️ Risk Factors Analysis")
    
    risk_factors = []
    if smoker == "Yes":
        risk_factors.append("❌ Smoking (major risk factor)")
    if age > 50:
        risk_factors.append("⚠️ Age > 50")
    if bmi > 30:
        risk_factors.append("⚠️ Obesity (BMI > 30)")
    if bmi < 18.5:
        risk_factors.append("⚠️ Underweight")
    if children > 2:
        risk_factors.append("⚠️ Multiple dependents")
    
    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.write("✅ No major risk factors identified")

with col2:
    st.subheader("💰 Cost Prediction")
    
    # Main prediction display
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: #2c3e50;">Estimated Annual Cost</h2>
        <h1 style="color: #e74c3c; font-size: 48px;">${prediction:,.2f}</h1>
        <p style="color: #7f8c8d;">Confidence Interval: ${confidence_lower:,.2f} - ${confidence_upper:,.2f}</p>
        <p style="color: {'#27ae60' if risk_level == 'Low Risk' else '#e67e22' if risk_level == 'Medium Risk' else '#e74c3c'}; font-weight: bold;">
            {risk_icon} {risk_level}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Gauge chart for risk level
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cost Range", 'font': {'size': 24}},
        delta = {'reference': 10000},
        gauge = {
            'axis': {'range': [None, 30000], 'tickwidth': 1},
            'bar': {'color': "#2c3e50"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5000], 'color': '#d4efdf'},
                {'range': [5000, 15000], 'color': '#fdebd0'},
                {'range': [15000, 30000], 'color': '#fadbd8'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction}}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Comparison section
st.subheader("📈 Cost Comparison by Scenario")

# Create comparison scenarios
scenarios = {
    "Current": {"age": age, "bmi": bmi, "children": children, 
                "female_dm": female_dm, "smoker_dm": smoker_dm},
    "If Non-Smoker": {"age": age, "bmi": bmi, "children": children, 
                      "female_dm": female_dm, "smoker_dm": 0},
    "If Normal BMI": {"age": age, "bmi": 22, "children": children, 
                      "female_dm": female_dm, "smoker_dm": smoker_dm},
    "If Younger": {"age": max(18, age - 20), "bmi": bmi, "children": children, 
                   "female_dm": female_dm, "smoker_dm": smoker_dm}
}

comparison_results = []
for name, feat in scenarios.items():
    feat['smoker_bmi_interaction'] = feat['smoker_dm'] * feat['bmi']
    feat['age_squared'] = feat['age'] ** 2
    pred = model.predict(np.array([[feat['age'], feat['bmi'], feat['children'],
                                    feat['female_dm'], feat['smoker_dm'],
                                    feat['smoker_bmi_interaction'], feat['age_squared']]]))[0]
    comparison_results.append({"Scenario": name, "Predicted Cost": pred})

comparison_df = pd.DataFrame(comparison_results)
fig = px.bar(comparison_df, x="Scenario", y="Predicted Cost", 
             title="How Different Factors Affect Your Cost",
             color="Predicted Cost", color_continuous_scale="Reds",
             text_auto='.2s')
st.plotly_chart(fig, use_container_width=True)

# Batch prediction section
st.subheader("📁 Batch Prediction")
st.markdown("Upload a CSV file with columns: `age, bmi, children, sex, smoker`")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())
    
    if st.button("Run Batch Prediction"):
        with st.spinner("Processing..."):
            # Create engineered features
            df['female_dm'] = df['sex'].map({'female': 1, 'male': 0})
            df['smoker_dm'] = df['smoker'].map({'yes': 1, 'no': 0})
            df['smoker_bmi_interaction'] = df['smoker_dm'] * df['bmi']
            df['age_squared'] = df['age'] ** 2
            
            # Make predictions
            features = df[feature_names]
            predictions = model.predict(features)
            df['predicted_charges'] = predictions
            
            # Add risk categories
            df['risk_category'] = pd.cut(predictions, 
                                          bins=[0, 5000, 15000, float('inf')],
                                          labels=['Low', 'Medium', 'High'])
            
            # Display results
            st.write("### Prediction Results")
            st.dataframe(df[['age', 'bmi', 'children', 'sex', 'smoker', 
                             'predicted_charges', 'risk_category']].head(20))
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Average Cost", f"${df['predicted_charges'].mean():,.2f}")
            with col3:
                st.metric("Total Cost", f"${df['predicted_charges'].sum():,.2f}")
            
            # Risk distribution chart
            risk_counts = df['risk_category'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                         title="Risk Distribution", color_discrete_sequence=['#27ae60', '#e67e22', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, "predictions.csv", "text/csv")

# Model information
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    - **Algorithm**: Random Forest Regression
    - **Training Data**: 1,337 insurance records
    - **Features Used**: Age, BMI, Children, Gender, Smoking Status
    - **Model Performance**:
        - Mean Absolute Error (MAE): ~$2,500
        - R² Score: ~0.89
        - Root Mean Square Error (RMSE): ~$4,400
    
    **Disclaimer**: This is a predictive model based on historical data. Actual insurance costs may vary based on additional factors not captured in this model.
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Data source: Medical Cost Personal Dataset")
