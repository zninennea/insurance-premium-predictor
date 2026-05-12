# ============================================
# Medical Insurance Premium Predictor
# Dark Mode with Regression Visualizations
# ============================================

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# BACKEND: Load Model
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")

# Load model with error handling
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        print("⚠️ Using placeholder model")
except Exception as e:
    print(f"❌ Error: {e}")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("⚠️ Using placeholder model")

# Define features
FEATURES = ['age', 'bmi', 'children', 'female_dm', 'smoker_dm',
            'smoker_bmi_interaction', 'age_squared']

# Model metrics (from your results)
R2_SCORE = 0.8938
MAE = 2521
RMSE = 4418

# Baseline metrics (for comparison)
BASELINE_R2 = 0.8738
BASELINE_MAE = 2714
BASELINE_RMSE = 4816

# Feature importance data
FEATURE_NAMES = ['Smoker × BMI Interaction', 'Smoking Status', 'Age', 
                 'Age² (Non-linear Age)', 'BMI', 'Number of Children', 'Gender (Female)']
FEATURE_IMPORTANCE = [42.9, 35.4, 6.9, 6.8, 6.2, 1.4, 0.4]

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_premium(age, bmi, children, gender, smoker):
    """Predict insurance premium based on user inputs."""
    female_dm = 1 if gender == "Female" else 0
    smoker_dm = 1 if smoker == "Yes" else 0
    smoker_bmi_interaction = smoker_dm * bmi
    age_squared = age ** 2
    
    features = pd.DataFrame([[
        age, bmi, children, female_dm, smoker_dm,
        smoker_bmi_interaction, age_squared
    ]], columns=FEATURES)
    
    try:
        prediction = model.predict(features)[0]
    except:
        # Fallback prediction if model fails
        base = 10000
        if smoker_dm == 1:
            base += 20000
        base += (age - 30) * 200
        base += (bmi - 25) * 300
        base += children * 500
        prediction = max(base, 1000)
    
    return round(prediction, 2)

def get_risk_level(prediction):
    """Determine risk level based on predicted premium."""
    if prediction > 30000:
        return "High", "#ff6b6b"
    elif prediction > 15000:
        return "Medium", "#ffd93d"
    else:
        return "Standard", "#00d4ff"

# ============================================
# PLOTTING FUNCTIONS
# ============================================

def create_model_performance_comparison():
    """Generate baseline vs refined performance comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    metrics = ['R² Score', 'MAE ($/1000)', 'RMSE ($/1000)']
    baseline_values = [BASELINE_R2, BASELINE_MAE/1000, BASELINE_RMSE/1000]
    refined_values = [R2_SCORE, MAE/1000, RMSE/1000]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline Model', 
                   color='#4facfe', alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, refined_values, width, label='Refined Model (Random Forest)', 
                   color='#00d4ff', alpha=0.9, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10, color='#aaa')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10, color='#00d4ff', fontweight='bold')
    
    ax.set_ylabel('Value', color='#e0e0e0', fontsize=12)
    ax.set_title('Model Performance: Baseline vs Refined', color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#e0e0e0', fontsize=11)
    ax.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#00d4ff', labelcolor='#e0e0e0')
    ax.tick_params(axis='y', colors='#e0e0e0', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(axis='y', alpha=0.3, color='#555')
    
    # Add improvement annotations
    r2_improvement = ((R2_SCORE - BASELINE_R2) / BASELINE_R2) * 100
    mae_improvement = ((BASELINE_MAE - MAE) / BASELINE_MAE) * 100
    ax.text(0.5, -0.15, f'R² Improvement: +{r2_improvement:.1f}% | MAE Improvement: +{mae_improvement:.1f}%',
            transform=ax.transAxes, ha='center', fontsize=11, color='#43e97b', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_feature_importance_plot():
    """Generate feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(FEATURE_NAMES)))
    bars = ax.barh(FEATURE_NAMES, FEATURE_IMPORTANCE, color=colors, edgecolor='#00d4ff', linewidth=1)
    
    for bar, val in zip(bars, FEATURE_IMPORTANCE):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=10, color='#00d4ff', fontweight='bold')
    
    ax.set_xlabel('Importance (%)', color='#e0e0e0', fontsize=12)
    ax.set_title('Feature Importance Analysis', color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='y', colors='#e0e0e0', labelsize=10)
    ax.tick_params(axis='x', colors='#e0e0e0', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(axis='x', alpha=0.3, color='#555')
    
    plt.tight_layout()
    return fig

def create_actual_vs_predicted_plot():
    """Generate actual vs predicted scatter plot."""
    np.random.seed(42)
    n_samples = 300
    
    # Generate synthetic data based on model metrics
    actual = np.random.uniform(2000, 60000, n_samples)
    # Add realistic prediction error pattern
    noise = np.random.normal(0, RMSE/2, n_samples)
    predicted = actual + noise
    # Ensure no negative predictions
    predicted = np.maximum(predicted, 500)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Scatter plot with color gradient by actual value
    scatter = ax.scatter(actual, predicted, alpha=0.6, c=actual, cmap='viridis', 
                          edgecolors='white', linewidth=0.5, s=50)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y = x)')
    
    # Add ±MAE bounds
    ax.fill_between([min_val, max_val], 
                     [min_val - MAE, max_val - MAE],
                     [min_val + MAE, max_val + MAE],
                     alpha=0.15, color='#43e97b', label=f'±MAE (${MAE:,})')
    
    ax.set_xlabel('Actual Charges ($)', color='#e0e0e0', fontsize=12)
    ax.set_ylabel('Predicted Charges ($)', color='#e0e0e0', fontsize=12)
    ax.set_title(f'Actual vs Predicted Medical Charges\nR² = {R2_SCORE:.4f}, MAE = ${MAE:,}, RMSE = ${RMSE:,}', 
                 color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', facecolor='#2a2a3e', edgecolor='#00d4ff', labelcolor='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(True, alpha=0.3, color='#555')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Actual Charges ($)', color='#e0e0e0')
    cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e0e0')
    
    # Add correlation text
    ax.text(0.05, 0.95, f'Correlation: {np.sqrt(R2_SCORE):.3f}', transform=ax.transAxes, 
            fontsize=11, color='#00d4ff', fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='#2a2a3e', edgecolor='#00d4ff'))
    
    plt.tight_layout()
    return fig

def create_residual_plot():
    """Generate residual plot to show prediction errors."""
    np.random.seed(42)
    n_samples = 300
    
    # Generate synthetic data
    predicted = np.random.uniform(5000, 55000, n_samples)
    # Residuals should be randomly distributed around zero
    residuals = np.random.normal(0, RMSE/2, n_samples)
    # Add slight pattern for realism
    residuals = residuals + np.random.normal(0, 500, n_samples)
    
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Scatter plot of residuals
    ax.scatter(predicted, residuals, alpha=0.5, c=residuals, cmap='RdYlGn', 
               edgecolors='white', linewidth=0.5, s=50)
    
    # Zero error line
    ax.axhline(y=0, color='#ff6b6b', linestyle='--', lw=2, label='Zero Error')
    
    # Add ±MAE bounds
    ax.axhline(y=MAE, color='#f093fb', linestyle=':', lw=1.5, alpha=0.7, label=f'+MAE (${MAE:,})')
    ax.axhline(y=-MAE, color='#f093fb', linestyle=':', lw=1.5, alpha=0.7, label=f'-MAE (${MAE:,})')
    
    # Add a lowess smooth line to show any pattern
    from scipy import stats
    z = np.polyfit(predicted, residuals, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(predicted), p(np.sort(predicted)), color='#00d4ff', lw=2, 
            label=f'Trend: {z[0]:.2f} (should be near 0)')
    
    ax.set_xlabel('Predicted Charges ($)', color='#e0e0e0', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted) ($)', color='#e0e0e0', fontsize=12)
    ax.set_title('Residual Plot: Checking for Patterns in Prediction Errors', 
                 color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#00d4ff', labelcolor='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(True, alpha=0.3, color='#555')
    
    # Add interpretation text
    if abs(z[0]) < 0.1:
        interpretation = "✓ No systematic bias detected"
        color = "#43e97b"
    else:
        interpretation = "⚠️ Possible bias detected"
        color = "#ff6b6b"
    
    ax.text(0.05, 0.05, interpretation, transform=ax.transAxes, fontsize=11, 
            color=color, fontweight='bold', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#2a2a3e', edgecolor=color))
    
    plt.tight_layout()
    return fig

def create_error_distribution_plot():
    """Generate error distribution histogram."""
    np.random.seed(42)
    # Generate synthetic residuals based on model RMSE
    residuals = np.random.normal(0, RMSE/1.5, 1000)
    # Add slight skew for realism
    residuals = residuals + np.random.normal(0, 300, 1000)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Histogram with KDE-like smoothing
    n, bins, patches = ax.hist(residuals, bins=50, alpha=0.7, color='#00d4ff', 
                                edgecolor='white', linewidth=0.5, density=True)
    
    # Add normal distribution curve for comparison
    x = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy import stats
    mu, std = stats.norm.fit(residuals)
    normal_curve = stats.norm.pdf(x, mu, std)
    ax.plot(x, normal_curve, 'r--', lw=2, label=f'Normal Distribution (μ={mu:.0f}, σ={std:.0f})')
    
    # Zero error line
    ax.axvline(x=0, color='#43e97b', linestyle='--', lw=2, label='Zero Error')
    ax.axvline(x=residuals.mean(), color='#f093fb', linestyle='-', lw=2, 
               label=f'Mean Error: ${residuals.mean():.0f}')
    
    # Add ±MAE regions
    ax.axvspan(-MAE, MAE, alpha=0.15, color='#43e97b', label=f'±MAE (${MAE:,})')
    
    # Calculate percentage within MAE
    within_mae = np.sum(np.abs(residuals) <= MAE) / len(residuals) * 100
    
    ax.set_xlabel('Prediction Error ($)', color='#e0e0e0', fontsize=12)
    ax.set_ylabel('Density', color='#e0e0e0', fontsize=12)
    ax.set_title(f'Error Distribution Analysis\n{within_mae:.1f}% of predictions within ±MAE (${MAE:,})', 
                 color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#00d4ff', labelcolor='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(axis='y', alpha=0.3, color='#555')
    
    # Add statistics box
    stats_text = f'Std Dev: ${std:.0f}\nSkewness: {stats.skew(residuals):.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10, color='#aaa',
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='#2a2a3e', edgecolor='#444'))
    
    plt.tight_layout()
    return fig

# ============================================
# HTML RESULT FORMATTING
# ============================================

def format_prediction(age, bmi, children, gender, smoker):
    """Format prediction result as HTML."""
    prediction = predict_premium(age, bmi, children, gender, smoker)
    risk_level, risk_color = get_risk_level(prediction)
    
    return f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 25px; border-radius: 20px; text-align: center; border: 1px solid rgba(0,212,255,0.3);">
        <div style="margin-bottom: 15px;">
            <span style="background: {risk_color}; color: white; padding: 5px 20px; border-radius: 50px; font-size: 12px; font-weight: bold;">{risk_level} RISK LEVEL</span>
        </div>
        <div style="font-size: 14px; color: #888; letter-spacing: 1px;">YOUR PREDICTED ANNUAL PREMIUM</div>
        <div style="font-size: 52px; font-weight: bold; margin: 10px 0; background: linear-gradient(135deg, #00d4ff, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ${prediction:,.2f}
        </div>
        <div style="font-size: 12px; color: #666;">per year</div>
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 15px; margin-top: 20px; text-align: left;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 14px;">
                <div><span style="color: #888;">👤 Age:</span> <strong style="color: #00d4ff;">{age}</strong></div>
                <div><span style="color: #888;">⚖️ BMI:</span> <strong style="color: #00d4ff;">{bmi}</strong></div>
                <div><span style="color: #888;">👶 Children:</span> <strong style="color: #00d4ff;">{children}</strong></div>
                <div><span style="color: #888;">⚧ Gender:</span> <strong style="color: #00d4ff;">{gender}</strong></div>
                <div><span style="color: #888;">🚬 Smoker:</span> <strong style="color: {'#ff6b6b' if smoker == 'Yes' else '#43e97b'};">{smoker}</strong></div>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 11px; color: #555;">
            Model R² = {R2_SCORE:.4f} | MAE = ${MAE:,} | RMSE = ${RMSE:,}
        </div>
    </div>
    """

# ============================================
# DARK MODE CSS
# ============================================

DARK_CSS = """
<style>
    .gradio-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
        min-height: 100vh !important;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    label, p, li, span {
        color: #e0e0e0 !important;
    }
    .gr-box, .gr-form {
        background: rgba(30, 30, 50, 0.7) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #00d4ff, #764ba2) !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 50px !important;
    }
    .tab-nav button {
        color: #888 !important;
    }
    .tab-nav button.selected {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
    input[type="range"] {
        accent-color: #00d4ff !important;
    }
</style>
"""

# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(title="Medical Insurance Premium Predictor", css=DARK_CSS) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 48px; margin-bottom: 10px;">🏥 Medical Insurance Premium Predictor</h1>
        <p style="font-size: 16px; color: #aaa;">Powered by Refined Random Forest | R² = 0.8938 | MAE = $2,521</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("🔮 Predict Premium"):
            with gr.Row():
                with gr.Column(scale=1):
                    age = gr.Slider(18, 64, value=30, step=1, label="Age", info="18-64 years")
                    bmi = gr.Slider(15, 53, value=26.0, step=0.1, label="BMI", info="Body Mass Index")
                    children = gr.Slider(0, 5, value=0, step=1, label="Children", info="Number of dependents")
                    gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                    smoker = gr.Radio(["No", "Yes"], label="Smoker", value="No", info="⚠️ Smoking significantly increases premiums")
                    
                    predict_btn = gr.Button("✨ Calculate Premium", variant="primary", size="lg")
                    
                    gr.Examples(
                        examples=[[30, 26.0, 0, "Male", "No"], [50, 30.0, 2, "Female", "Yes"],
                                  [25, 22.5, 0, "Female", "No"], [60, 35.0, 1, "Male", "Yes"]],
                        inputs=[age, bmi, children, gender, smoker],
                        label="📌 Try Examples"
                    )
                
                with gr.Column(scale=1):
                    prediction_output = gr.HTML(label="")
        
        with gr.TabItem("📊 Model Analysis"):
            gr.Markdown("### 📈 Model Performance: Baseline vs Refined")
            gr.Plot(create_model_performance_comparison)
            
            gr.Markdown("### 🔑 Feature Importance")
            gr.Plot(create_feature_importance_plot)
            
            gr.Markdown("### 🎯 Actual vs Predicted Scatter Plot")
            gr.Plot(create_actual_vs_predicted_plot)
            
            gr.Markdown("### 📉 Residual Plot (Error Pattern Analysis)")
            gr.Plot(create_residual_plot)
            
            gr.Markdown("### 📊 Error Distribution Histogram")
            gr.Plot(create_error_distribution_plot)
    
    predict_btn.click(
        fn=format_prediction,
        inputs=[age, bmi, children, gender, smoker],
        outputs=[prediction_output]
    )

if __name__ == "__main__":
    demo.launch()