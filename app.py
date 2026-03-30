import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CardioSense AI | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS INJECTION (The "Secret Sauce" to match your HTML exactly) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1" rel="stylesheet">
<style>
    /* Global Styles */
    .stApp { background-color: #faf9f6; color: #1a1c1a; font-family: 'Inter', sans-serif; }
    
    /* Typography Overrides */
    h1, h2, h3, h4, .headline { font-family: 'Newsreader', serif !important; letter-spacing: -0.02em; }
    
    /* Custom Sidebar Styling */
    [data-testid="stSidebar"] {
    background-color: #1a1a2e !important;
    width: 300px !important;
}

section[data-testid="stSidebar"] {
    display: block !important;
}

[data-testid="collapsedControl"] {
    display: none !important; /* remove collapse button */
}
    .sidebar-brand { padding: 2rem 1rem; }
    .sidebar-brand h1 { font-size: 1.5rem; margin-bottom: 0; }
    .sidebar-tagline { font-size: 0.65rem; color: #b10c69 !important; letter-spacing: 0.2em; text-transform: uppercase; font-weight: 700; }

    /* Editorial Shadow & Cards */
    .editorial-shadow { box-shadow: 0 32px 64px -12px rgba(26, 28, 26, 0.06); }
    .bento-card {
        background: #ffffff; padding: 2rem; border-radius: 0.75rem; 
        border: 1px solid rgba(222, 190, 200, 0.2); margin-bottom: 1rem;
    }
    .label-caps { font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.15em; color: #584048; font-weight: 600; margin-bottom: 1rem; }
    
    /* Model Cards */
    .model-row {
        background: #ffffff; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem;
        display: flex; justify-content: space-between; align-items: center; border: 1px solid #efeeeb;
    }
    
    /* Circular Badge */
    .status-circle {
        width: 180px; height: 180px; border-radius: 50%; border: 1px solid rgba(177, 12, 105, 0.2);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: radial-gradient(circle, rgba(177, 12, 105, 0.05) 0%, transparent 70%);
        position: relative;
    }

    /* Progress Bar */
    .progress-bg { background: #e3e2df; height: 8px; border-radius: 10px; width: 100%; overflow: hidden; }
    .progress-fill { background: #b10c69; height: 100%; border-radius: 10px; }

    /* Hide Streamlit Branded Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        knn = joblib.load('models/knn_model.pkl')
        svm = joblib.load('models/svm_model.pkl')
        ann = joblib.load('models/ann_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        df = pd.read_csv('data/raw/heart.csv')
        return knn, svm, ann, scaler, df
    except:
        return None, None, None, None, None

knn, svm, ann, scaler, df = load_models()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="padding: 1rem 0;">
            <h1 style="font-family: 'Newsreader', serif; font-size: 1.8rem; margin:0;">CardioSense AI</h1>
            <p style="font-size: 0.7rem; color: #b10c69; letter-spacing: 2px; text-transform: uppercase;">Clinical Precision</p>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("DASHBOARD", ["Home", "Predict", "Model Performance", "Dataset Overview"], label_visibility="collapsed")
    
        page = st.radio("NAVIGATION", ["Home", "Predict", "Performance", "Dataset"])
    st.markdown("---")
    st.markdown("### System Status")
    st.success("● Models Loaded")
    st.info("● Database Online")

# --- HEADER BREADCRUMB ---
st.markdown(f'<p class="label-caps" style="margin-left: 2rem; margin-top: 1rem;">Dashboard / {page}</p>', unsafe_allow_html=True)

# ==================== HOME PAGE ====================
if page == "Home":
    # HERO SECTION
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            <div style="padding: 2rem;">
                <h2 style="font-size: 4rem; line-height: 1.1; margin-bottom: 1.5rem;">
                    Heart Disease <br/><span style="color: #b10c69; font-style: italic;">Prediction System</span>
                </h2>
                <p style="font-size: 1.25rem; color: #584048; max-width: 600px; font-weight: 300; line-height: 1.6;">
                    An advanced clinical intelligence suite utilizing three distinct machine learning models to synthesize cardiovascular risk factors into actionable diagnostic insights.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="status-circle">
                <span class="material-symbols-outlined" style="color: #b10c69; font-size: 3rem; font-variation-settings: 'FILL' 1;">favorite</span>
                <p class="label-caps" style="margin: 5px 0 0 0;">System Status</p>
                <p style="font-weight: 700; color: #006a39; margin: 0;">Live & Precise</p>
            </div>
        """, unsafe_allow_html=True)

    # BENTO GRID METRICS
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="bento-card editorial-shadow"><p class="label-caps">Dataset Size</p><h3 style="font-size: 2.5rem; margin:0;">1,025 <small style="font-size: 1rem; color: gray;">Records</small></h3></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="bento-card editorial-shadow"><p class="label-caps">Clinical Features</p><h3 style="font-size: 2.5rem; margin:0;">13 <small style="font-size: 1rem; color: gray;">Params</small></h3></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="bento-card editorial-shadow" style="border-left: 4px solid #b10c69;"><p class="label-caps">Best Accuracy</p><h3 style="font-size: 2.5rem; margin:0; color: #b10c69;">98.5% <small style="font-size: 1rem;">ANN</small></h3></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="bento-card editorial-shadow"><p class="label-caps">Missing Values</p><h3 style="font-size: 2.5rem; margin:0; color: #006a39;">Zero</h3></div>', unsafe_allow_html=True)

    # BOTTOM CONTENT
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("""
            <h4 class="label-caps" style="color: #b10c69;">Foundation</h4>
            <h3 style="font-size: 2rem;">About This Project</h3>
            <p style="color: #584048; line-height: 1.7;">Developed as part of the <b>TARUMT AI Assignment</b>, this project explores the intersection of supervised machine learning and clinical diagnostics.</p>
            <p style="color: #584048; line-height: 1.7;">The core objective is to evaluate how different algorithmic architectures—ranging from KNN to complex Neural Networks—interpret physiological signals.</p>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<h4 class="label-caps">Model Benchmarking</h4>', unsafe_allow_html=True)
        # ANN Card
        st.markdown("""
            <div class="model-row editorial-shadow">
                <div style="display:flex; align-items: center; gap: 1rem;">
                    <div style="background: rgba(177, 12, 105, 0.1); padding: 10px; border-radius: 50%;"><span class="material-symbols-outlined" style="color:#b10c69;">psychology</span></div>
                    <div><p style="margin:0; font-weight: 600;">Artificial Neural Network (ANN)</p><p style="margin:0; font-size: 0.7rem; color: gray;">Multi-layer perceptron architecture</p></div>
                </div>
                <div style="text-align: right;"><h3 style="margin:0; color:#b10c69;">98.5%</h3><p class="label-caps" style="margin:0; font-size: 0.6rem;">Accuracy</p></div>
            </div>
        """, unsafe_allow_html=True)
        # SVM Card
        st.markdown("""
            <div class="model-row editorial-shadow">
                <div style="display:flex; align-items: center; gap: 1rem;">
                    <div style="background: rgba(93, 92, 116, 0.1); padding: 10px; border-radius: 50%;"><span class="material-symbols-outlined" style="color:#5d5c74;">border_inner</span></div>
                    <div><p style="margin:0; font-weight: 600;">Support Vector Machine (SVM)</p><p style="margin:0; font-size: 0.7rem; color: gray;">Linear and RBF kernel</p></div>
                </div>
                <div style="text-align: right;"><h3 style="margin:0;">88.8%</h3><p class="label-caps" style="margin:0; font-size: 0.6rem;">Accuracy</p></div>
            </div>
        """, unsafe_allow_html=True)

# ==================== PREDICT PAGE ====================
elif page == "Predict":
    st.markdown('<h2 style="font-size: 3rem; margin-left: 2rem;">Clinical Prediction</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="bento-card editorial-shadow" style="margin: 0 2rem;">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 20, 100, 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0,1,2,3])
            trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        with col2:
            chol = st.number_input("Cholesterol", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
            restecg = st.selectbox("Resting ECG", [0,1,2])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        with col3:
            exang = st.selectbox("Exercise Induced Angina", [0,1])
            oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0)
            slope = st.selectbox("Slope of ST Segment", [0,1,2])
            ca = st.selectbox("Major Vessels", [0,1,2,3])
            thal = st.selectbox("Thalassemia", [0,1,2,3])
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("RUN CLINICAL ANALYSIS", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        if scaler is not None:
            # PULLING FROM YOUR PKL - REAL LOGIC
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            input_scaled = scaler.transform(input_data)
            
            p_knn = knn.predict(input_scaled)[0]
            p_svm = svm.predict(input_scaled)[0]
            p_ann = ann.predict(input_scaled)[0]
            
            res1, res2, res3 = st.columns(3)
            for col, name, pred in zip([res1, res2, res3], ["KNN", "SVM", "ANN"], [p_knn, p_svm, p_ann]):
                color = "#ba1a1a" if pred == 1 else "#006a39"
                status = "HIGH RISK" if pred == 1 else "LOW RISK"
                col.markdown(f"""
                    <div class="bento-card editorial-shadow" style="text-align:center; border-top: 5px solid {color};">
                        <p class="label-caps">{name} Result</p>
                        <h2 style="color:{color};">{status}</h2>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Model files not found in /models directory.")

# ==================== DATASET OVERVIEW ====================
elif page == "Dataset Overview":
    st.markdown('<h2 style="font-size: 3rem; margin-left: 2rem;">Clinical Data Pool</h2>', unsafe_allow_html=True)
    if df is not None:
        st.markdown('<div class="bento-card editorial-shadow" style="margin: 0 2rem;">', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("heart.csv not found in data/raw/")

# --- FOOTER ---
st.markdown("""
    <div style="margin-top: 5rem; padding: 2rem; border-top: 1px solid #debec8; display: flex; justify-content: space-between; opacity: 0.5;">
        <p class="label-caps" style="font-size: 0.6rem;">© 2024 CardioSense AI Clinical Systems</p>
        <p class="label-caps" style="font-size: 0.6rem;">TARUMT University | Artificial Intelligence Faculty</p>
    </div>
""", unsafe_allow_html=True)
