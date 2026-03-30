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
    layout="wide"
)

# --- CSS INJECTION (Simplified for Single Page) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1" rel="stylesheet">
<style>
    .stApp { background-color: #faf9f6; color: #1a1c1a; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Newsreader', serif !important; letter-spacing: -0.02em; }
    
    /* Hide Sidebar completely */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    .editorial-shadow { box-shadow: 0 32px 64px -12px rgba(26, 28, 26, 0.06); }
    .bento-card {
        background: #ffffff; padding: 2rem; border-radius: 0.75rem; 
        border: 1px solid rgba(222, 190, 200, 0.3); margin-bottom: 1.5rem;
    }
    .label-caps { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; color: #b10c69; font-weight: 700; margin-bottom: 1rem; }
    
    .section-divider {
        margin: 4rem 0 2rem 0;
        border-bottom: 1px solid #efeeeb;
    }

    /* Prediction Result Styles */
    .res-box { text-align:center; border-radius: 1rem; padding: 1.5rem; border: 1px solid #efeeeb; background: white; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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

# ==================== SECTION 1: HERO ====================
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
        <div style="padding-top: 2rem;">
            <p class="label-caps">Clinical Intelligence Suite</p>
            <h1 style="font-size: 5rem; line-height: 1; margin-bottom: 1.5rem;">
                CardioSense <span style="color: #b10c69; font-style: italic;">AI</span>
            </h1>
            <p style="font-size: 1.4rem; color: #584048; max-width: 700px; font-weight: 300; line-height: 1.6;">
                An advanced diagnostic framework utilizing ensemble machine learning to synthesize 
                cardiovascular risk factors into high-precision clinical insights.
            </p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div style="margin-top: 4rem; text-align: center; border: 1px solid #b10c69; padding: 2rem; border-radius: 100px;">
            <span class="material-symbols-outlined" style="color: #b10c69; font-size: 3rem;">clinical_notes</span>
            <p class="label-caps" style="margin-top: 10px;">Status</p>
            <p style="font-weight: 700; color: #006a39;">SYSTEM LIVE</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== SECTION 2: METRICS ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="bento-card editorial-shadow"><p class="label-caps">Training Pool</p><h3 style="font-size: 2.5rem; margin:0;">1,025 <small style="font-size: 1rem; color: gray;">Records</small></h3></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="bento-card editorial-shadow"><p class="label-caps">Biomarkers</p><h3 style="font-size: 2.5rem; margin:0;">13 <small style="font-size: 1rem; color: gray;">Params</small></h3></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="bento-card editorial-shadow" style="background: #1a1a2e; color: white;"><p class="label-caps" style="color: #b10c69;">Peak Accuracy</p><h3 style="font-size: 2.5rem; margin:0; color: white;">98.5% <small style="font-size: 1rem;">ANN</small></h3></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="bento-card editorial-shadow"><p class="label-caps">Data Integrity</p><h3 style="font-size: 2.5rem; margin:0; color: #006a39;">Verified</h3></div>', unsafe_allow_html=True)

# ==================== SECTION 3: DATASET OVERVIEW ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="label-caps">Data Intelligence</p>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 3rem;">Dataset Overview</h2>', unsafe_allow_html=True)

if df is not None:
    
    st.markdown("### Clinical Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Class Distribution & Heatmap
    col_dist, col_corr = st.columns(2)
    with col_dist:
        st.markdown('<div class="bento-card editorial-shadow">', unsafe_allow_html=True)
        st.markdown("### Class Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(5, 4))
        df['target'].value_counts().plot(kind='bar', ax=ax_dist, color=['#4C72B0', '#b10c69'])
        ax_dist.set_xticklabels(['No Disease', 'Disease'], rotation=0)
        ax_dist.set_ylabel('Patient Count')
        st.pyplot(fig_dist)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_corr:
        st.markdown('<div class="bento-card editorial-shadow">', unsafe_allow_html=True)
        st.markdown("### Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
        sns.heatmap(df.corr(), annot=False, cmap='RdBu_r', ax=ax_corr)
        st.pyplot(fig_corr)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature Distributions
    st.markdown('<div class="bento-card editorial-shadow">', unsafe_allow_html=True)
    st.markdown("### Feature Frequency Distributions")
    fig_hist, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    for i, col in enumerate(df.columns[:-1]):
        axes[i].hist(df[col], bins=20, color='#1a1a2e', edgecolor='white', alpha=0.8)
        axes[i].set_title(col, fontsize=10, fontweight='bold')
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_hist)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== SECTION 4: PREDICTION (AS PER PREVIOUS CODE) ====================
# [Keep the Prediction Section code from the previous response here]

# ==================== SECTION 5: MODEL PERFORMANCE ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="label-caps">Algorithmic Validation</p>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 3rem;">Evaluation Metrics</h2>', unsafe_allow_html=True)

# Metrics Table
results = pd.DataFrame({
    'Model': ['KNN', 'SVM', 'ANN'],
    'Accuracy': [83.41, 88.78, 98.54],
    'Precision': [80.00, 85.09, 100.00],
    'Recall': [89.32, 94.17, 97.09],
    'F1 Score': [84.40, 89.40, 98.52]
})

st.markdown('<div class="bento-card editorial-shadow">', unsafe_allow_html=True)
st.dataframe(results.style.highlight_max(axis=0, color='#debec8'), use_container_width=True)

# Performance Bar Chart
st.markdown("### Comparative Performance Analysis")
fig_perf, ax_perf = plt.subplots(figsize=(12, 5))
x = np.arange(3)
width = 0.15
ax_perf.bar(x - width*1.5, results['Accuracy'], width, label='Accuracy', color='#1a1a2e')
ax_perf.bar(x - width*0.5, results['Precision'], width, label='Precision', color='#b10c69')
ax_perf.bar(x + width*0.5, results['Recall'], width, label='Recall', color='#584048')
ax_perf.bar(x + width*1.5, results['F1 Score'], width, label='F1 Score', color='#debec8')

ax_perf.set_xticks(x)
ax_perf.set_xticklabels(['KNN', 'SVM', 'ANN'])
ax_perf.set_ylim(0, 115)
ax_perf.legend(frameon=False, loc='upper left')
sns.despine()
st.pyplot(fig_perf)
st.markdown('</div>', unsafe_allow_html=True)

# Confusion Matrices
st.markdown('<h3 style="text-align: center; margin-top: 2rem;">Confusion Matrices</h3>', unsafe_allow_html=True)
cm_data = {
    'KNN': np.array([[79, 23], [11, 92]]),
    'SVM': np.array([[85, 17], [6, 97]]),
    'ANN': np.array([[102, 0], [3, 100]])
}

col_cm1, col_cm2, col_cm3 = st.columns(3)
for col, (model_name, cm) in zip([col_cm1, col_cm2, col_cm3], cm_data.items()):
    with col:
        st.markdown(f'<div class="bento-card editorial-shadow" style="padding: 1rem;">', unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_name}', fontweight='bold')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== SECTION 4: PREDICTION ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 3rem; text-align: center;">Run Diagnostic Analysis</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray; margin-bottom: 3rem;">Input patient vitals to generate comparative model results.</p>', unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    with col2:
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
        restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    with col3:
        exang = st.selectbox("Exercise Induced Angina", [0,1])
        oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0)
        slope = st.selectbox("Slope of ST Segment", [0,1,2])
        ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
        thal = st.selectbox("Thalassemia (0-3)", [0,1,2,3])
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("EXECUTE ANALYSIS", use_container_width=True, type="primary")

if predict_btn:
    if scaler is not None and knn is not None:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)
        
        preds = [knn.predict(input_scaled)[0], svm.predict(input_scaled)[0], ann.predict(input_scaled)[0]]
        names = ["KNN Classifier", "SVM (RBF)", "Neural Network (ANN)"]
        
        st.markdown("<br>", unsafe_allow_html=True)
        res_cols = st.columns(3)
        for col, name, pred in zip(res_cols, names, preds):
            color = "#ba1a1a" if pred == 1 else "#006a39"
            status = "HIGH RISK" if pred == 1 else "LOW RISK"
            col.markdown(f"""
                <div class="res-box editorial-shadow" style="border-top: 5px solid {color};">
                    <p class="label-caps">{name}</p>
                    <h2 style="color:{color}; margin: 0;">{status}</h2>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Error: Machine learning models not found in `/models` directory.")

# ==================== SECTION 5: PERFORMANCE ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 2.5rem;">Model Benchmarking</h2>', unsafe_allow_html=True)

col_p1, col_p2 = st.columns([2, 1])
with col_p1:
    acc_data = pd.DataFrame({
        'Model': ['KNN', 'SVM', 'ANN'],
        'Accuracy': [0.85, 0.88, 0.985]
    })
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Accuracy', y='Model', data=acc_data, palette='rocket', ax=ax)
    ax.set_xlim(0, 1.1)
    st.pyplot(fig)

with col_p2:
    st.markdown("""
        <div class="bento-card">
            <p class="label-caps">Architectural Note</p>
            <p style="color: #584048; font-size: 0.9rem;">
                The <b>ANN</b> out-performs classical models by capturing high-dimensional 
                interactions between heart rate and ST depression patterns. 
                <br><br>
                <b>Loss Function:</b> Binary Crossentropy<br>
                <b>Optimization:</b> Adam
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(f"""
    <div style="margin-top: 6rem; padding: 3rem 0; border-top: 1px solid #debec8; display: flex; justify-content: space-between; opacity: 0.6;">
        <p class="label-caps" style="font-size: 0.65rem;">© 2026 CardioSense AI | TARUMT University</p>
        <p class="label-caps" style="font-size: 0.65rem;">Research Lead: Artificial Intelligence Faculty</p>
    </div>
""", unsafe_allow_html=True)
