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

# --- CLEANED CSS ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Newsreader:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
    .stApp { background-color: #faf9f6; color: #1a1c1a; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Newsreader', serif !important; }
    
    /* Hide Sidebar */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

    /* Layout Components */
    .bento-card {
        background: #ffffff; padding: 2rem; border-radius: 0.75rem; 
        border: 1px solid #efeeeb; margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    .label-caps { 
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.15em; 
        color: #b10c69; font-weight: 700; margin-bottom: 0.5rem; 
    }
    
    .section-divider {
        margin: 3rem 0; border-bottom: 1px solid #debec8; opacity: 0.3;
    }

    .res-box { 
        text-align:center; border-radius: 1rem; padding: 1.5rem; 
        background: white; border: 1px solid #efeeeb;
    }

    /* Clean up Streamlit padding */
    .block-container { padding-top: 2rem !important; }
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
st.markdown('<p class="label-caps">Clinical Intelligence Suite</p>', unsafe_allow_html=True)
st.markdown('<h1 style="font-size: 4.5rem; line-height: 1; margin: 0;">CardioSense <span style="color: #b10c69; font-style: italic;">AI</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.2rem; color: #584048; max-width: 800px; margin-top: 1rem;">Supervised machine learning framework for cardiovascular risk assessment.</p>', unsafe_allow_html=True)

# ==================== SECTION 2: TOP METRICS ====================
st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
with m1: st.markdown('<div class="bento-card"><p class="label-caps">Records</p><h2 style="margin:0;">1,025</h2></div>', unsafe_allow_html=True)
with m2: st.markdown('<div class="bento-card"><p class="label-caps">Biomarkers</p><h2 style="margin:0;">13</h2></div>', unsafe_allow_html=True)
with m3: st.markdown('<div class="bento-card" style="background:#1a1a2e; color:white;"><p class="label-caps">Best Accuracy</p><h2 style="margin:0;">98.5%</h2></div>', unsafe_allow_html=True)
with m4: st.markdown('<div class="bento-card"><p class="label-caps">Status</p><h2 style="margin:0; color:#006a39;">Live</h2></div>', unsafe_allow_html=True)

# ==================== SECTION 3: DATASET OVERVIEW ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 2.5rem;">Clinical Data Pool</h2>', unsafe_allow_html=True)

if df is not None:
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown('### Sample Records')
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown('### Class Balance')
        fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
        df['target'].value_counts().plot(kind='bar', ax=ax_dist, color=['#1a1a2e', '#b10c69'])
        ax_dist.set_xticklabels(['No Disease', 'Disease'], rotation=0)
        st.pyplot(fig_dist)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.markdown('### Feature Correlation')
        fig_corr, ax_corr = plt.subplots(figsize=(5, 3))
        sns.heatmap(df.corr(), annot=False, cmap='PuRd', ax=ax_corr)
        st.pyplot(fig_corr)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    st.markdown('### Feature Distributions')
    fig_hist, axes = plt.subplots(2, 7, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(df.columns[:-1]):
        axes[i].hist(df[col], bins=15, color='#584048', edgecolor='white')
        axes[i].set_title(col, fontsize=9)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_hist)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== SECTION 4: PREDICTION ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 2.5rem; text-align:center;">Predictive Diagnostic</h2>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="bento-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])
        trestbps = st.number_input("Blood Pressure", 80, 200, 120)
    with c2:
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Sugar > 120", [0,1])
        restecg = st.selectbox("Resting ECG", [0,1,2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    with c3:
        exang = st.selectbox("Exercise Angina", [0,1])
        oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0)
        slope = st.selectbox("ST Slope", [0,1,2])
        ca = st.selectbox("Major Vessels", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [0,1,2,3])
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("RUN ANALYSIS", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:
    if scaler is not None and knn is not None:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)
        
        preds = [knn.predict(input_scaled)[0], svm.predict(input_scaled)[0], ann.predict(input_scaled)[0]]
        names = ["KNN", "SVM", "ANN"]
        
        res_cols = st.columns(3)
        for col, name, pred in zip(res_cols, names, preds):
            color = "#ba1a1a" if pred == 1 else "#006a39"
            status = "POSITIVE" if pred == 1 else "NEGATIVE"
            col.markdown(f"""
                <div class="res-box" style="border-top: 4px solid {color};">
                    <p class="label-caps">{name} Result</p>
                    <h2 style="color:{color}; margin: 0;">{status}</h2>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Model files missing in /models")

# ==================== SECTION 5: MODEL PERFORMANCE ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 2.5rem;">Performance Benchmarking</h2>', unsafe_allow_html=True)

st.markdown('<div class="bento-card">', unsafe_allow_html=True)
results = pd.DataFrame({
    'Model': ['KNN', 'SVM', 'ANN'],
    'Accuracy': [83.41, 88.78, 98.54],
    'Precision': [80.00, 85.09, 100.00],
    'Recall': [89.32, 94.17, 97.09],
    'F1 Score': [84.40, 89.40, 98.52]
})
st.dataframe(results, use_container_width=True)

fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
results.set_index('Model').plot(kind='bar', ax=ax_bar, color=['#1a1a2e', '#b10c69', '#584048', '#debec8'])
plt.xticks(rotation=0)
st.pyplot(fig_bar)
st.markdown('</div>', unsafe_allow_html=True)

# Confusion Matrices
st.markdown('### Confusion Matrices')
col_cm1, col_cm2, col_cm3 = st.columns(3)
cm_data = {
    'KNN': np.array([[79, 23], [11, 92]]),
    'SVM': np.array([[85, 17], [6, 97]]),
    'ANN': np.array([[102, 0], [3, 100]])
}

for col, (model_name, cm) in zip([col_cm1, col_cm2, col_cm3], cm_data.items()):
    with col:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd', ax=ax_cm, cbar=False)
        ax_cm.set_title(model_name)
        st.pyplot(fig_cm)
        st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; opacity:0.6; font-size:0.8rem;">© 2026 CardioSense AI | TARUMT University</p>', unsafe_allow_html=True)
