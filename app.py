import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inject Custom CSS & Fonts ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    .main {
        background-color: #faf9f6;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, .headline {
        font-family: 'Newsreader', serif !important;
    }
    
    /* Editorial Shadow & Panels */
    .editorial-card {
        background: white;
        padding: 2rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(222, 190, 200, 0.3);
        box-shadow: 0 32px 64px -12px rgba(26, 28, 26, 0.06);
        margin-bottom: 1.5rem;
    }
    
    /* Metrics Bento Grid */
    .metric-box {
        padding: 1.5rem;
        background: #ffffff;
        border-radius: 0.75rem;
        border-left: 4px solid #b10c69;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #584048;
        font-weight: 700;
    }
    .metric-value {
        font-family: 'Newsreader', serif;
        font-size: 2.5rem;
        color: #1a1c1a;
    }

    /* Model Cards */
    .model-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem;
        background: white;
        border-radius: 12px;
        border: 1px solid #efeeeb;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
    }
    
    /* Custom Sidebar override */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #b10c69, #d33182);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Models & Data ---
@st.cache_resource
def load_resources():
    knn = joblib.load('models/knn_model.pkl')
    svm = joblib.load('models/svm_model.pkl')
    ann = joblib.load('models/ann_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    df = pd.read_csv('data/raw/heart.csv')
    return knn, svm, ann, scaler, df

try:
    knn, svm, ann, scaler, df = load_resources()
except:
    st.error("Model files not found. Please ensure models/ and data/ directories are set up.")
    st.stop()

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
        <div style="padding: 1rem 0;">
            <h1 style="font-family: 'Newsreader', serif; font-size: 1.8rem; margin:0;">CardioSense AI</h1>
            <p style="font-size: 0.7rem; color: #b10c69; letter-spacing: 2px; text-transform: uppercase;">Clinical Precision</p>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("NAVIGATION", ["Home", "Predict", "Performance", "Dataset"])
    st.markdown("---")
    st.markdown("### System Status")
    st.success("● Models Loaded")
    st.info("● Database Online")

# ==================== HOME ====================
if page == "Home":
    # Hero Section
    st.markdown("""
        <div style="margin-top: 2rem; margin-bottom: 3rem;">
            <h2 class="headline" style="font-size: 4rem; line-height: 1; color: #1a1c1a;">
                Heart Disease <br/><span style="color: #b10c69; font-style: italic;">Prediction System</span>
            </h2>
            <p style="font-size: 1.2rem; color: #584048; max-width: 600px; font-weight: 300; margin-top: 1.5rem;">
                An advanced clinical intelligence suite utilizing machine learning architectures to synthesize cardiovascular risk factors into diagnostic insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Bento Grid Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-box"><p class="metric-label">Dataset Size</p><p class="metric-value">1,025</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-box"><p class="metric-label">Features</p><p class="metric-value">13</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-box" style="border-left-color: #b10c69;"><p class="metric-label">Best Accuracy</p><p class="metric-value" style="color:#b10c69">98.5%</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-box"><p class="metric-label">Status</p><p class="metric-value" style="color:#006a39">Live</p></div>', unsafe_allow_html=True)

    # Two Column Layout
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
            <h4 class="metric-label" style="color: #b10c69;">Foundation</h4>
            <h3 class="headline" style="font-size: 2rem;">About Project</h3>
            <p style="color: #584048; line-height: 1.6;">Developed as part of the <b>TARUMT AI Assignment</b>, this project explores supervised machine learning in clinical diagnostics.</p>
            <p style="color: #584048; line-height: 1.6;">The core objective is to evaluate how architectures—from instance-based learning (KNN) to complex neural structures (ANN)—interpret physiological signals.</p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<h4 class="metric-label">Model Benchmarking</h4>', unsafe_allow_html=True)
        # Model performance rows
        models = [
            ("Artificial Neural Network (ANN)", "98.5%", "psychology", "#b10c69"),
            ("Support Vector Machine (SVM)", "88.8%", "border_inner", "#5d5c74"),
            ("K-Nearest Neighbors (KNN)", "83.4%", "hub", "#584048")
        ]
        for name, acc, icon, color in models:
            st.markdown(f"""
                <div class="model-card" style="margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span class="material-symbols-outlined" style="color: {color}; font-size: 2rem;">{icon}</span>
                        <div>
                            <p style="margin:0; font-weight: 600;">{name}</p>
                            <p style="margin:0; font-size: 0.7rem; color: gray;">Accuracy Rate</p>
                        </div>
                    </div>
                    <div style="font-family: 'Newsreader', serif; font-size: 1.8rem; color: {color};">{acc}</div>
                </div>
            """, unsafe_allow_html=True)

# ==================== PREDICT ====================
elif page == "Predict":
    st.markdown('<h2 class="headline" style="font-size: 3rem;">Clinical Analysis</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="editorial-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", 20, 100, 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0,1,2,3],
                            format_func=lambda x: ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"][x])
            trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
            
        with col2:
            st.markdown("**Clinical Tests**")
            chol = st.number_input("Cholesterol", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG", [0,1,2], format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)

        with col3:
            st.markdown("**Advanced Features**")
            exang = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0)
            slope = st.selectbox("Slope of ST Segment", [0,1,2])
            ca = st.selectbox("Major Vessels", [0,1,2,3])
            thal = st.selectbox("Thalassemia", [0,1,2,3])

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Generate Diagnostic Report", use_container_width=True):
        input_data = scaler.transform([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        preds = {
            "KNN": knn.predict(input_data)[0],
            "SVM": svm.predict(input_data)[0],
            "ANN": ann.predict(input_data)[0]
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2, res_col3 = st.columns(3)
        
        for i, (m_name, p) in enumerate(preds.items()):
            with [res_col1, res_col2, res_col3][i]:
                color = "#ba1a1a" if p == 1 else "#006a39"
                label = "HIGH RISK" if p == 1 else "LOW RISK"
                st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: white; border-top: 5px solid {color}; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.05);">
                        <p class="metric-label">{m_name} Prediction</p>
                        <h3 style="color: {color}; margin-top: 0.5rem;">{label}</h3>
                    </div>
                """, unsafe_allow_html=True)

# ==================== PERFORMANCE / DATASET ====================
# (Simplified for length - Use the same wrapping logic as above)
elif page == "Performance":
    st.markdown('<h2 class="headline" style="font-size: 3rem;">Model Metrics</h2>', unsafe_allow_html=True)
    # Re-use the Bar Chart logic from your original code but place inside a st.markdown div
    results = pd.DataFrame({
        'Model': ['KNN', 'SVM', 'ANN'],
        'Accuracy': [83.41, 88.78, 98.54],
        'Precision': [80.00, 85.09, 100.00],
        'Recall': [89.32, 94.17, 97.09]
    })
    st.dataframe(results, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=results.melt(id_vars='Model'), x='Model', y='value', hue='variable', palette='magma', ax=ax)
    ax.set_facecolor('#faf9f6')
    fig.patch.set_facecolor('#faf9f6')
    st.pyplot(fig)

elif page == "Dataset":
    st.markdown('<h2 class="headline" style="font-size: 3rem;">Dataset Overview</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

# Footer
st.markdown("""
    <div style="margin-top: 5rem; padding-top: 2rem; border-top: 1px solid #debec8; opacity: 0.5; display: flex; justify-content: space-between;">
        <p style="font-size: 0.6rem; text-transform: uppercase;">© 2024 CardioSense AI Clinical Systems</p>
        <p style="font-size: 0.6rem; text-transform: uppercase;">TARUMT Artificial Intelligence Faculty</p>
    </div>
""", unsafe_allow_html=True)
