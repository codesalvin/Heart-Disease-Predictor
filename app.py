import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #ffffff !important;
    }

    .main, .block-container {
        background-color: #ffffff !important;
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #1a1a2e !important;
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stRadio {
        display: none !important;
    }

    .sidebar-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 26px;
        color: white;
        margin-bottom: 4px;
        padding: 0 1.5rem;
    }

    .sidebar-tagline {
        font-size: 12px;
        color: #888 !important;
        margin-bottom: 40px;
        padding: 0 1.5rem;
    }

    .nav-divider {
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 16px 24px;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        border: 1px solid #f0ede8;
        margin-bottom: 16px;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 42px;
        color: #1a1a2e;
        line-height: 1;
        margin: 8px 0;
    }

    .metric-label {
        font-size: 11px;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }

    .metric-sub {
        font-size: 13px;
        color: #888;
        margin-top: 4px;
    }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 52px;
        color: #1a1a2e;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .hero-sub {
        font-size: 16px;
        color: #aaa;
        font-weight: 300;
        margin-bottom: 32px;
    }

    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 28px;
        color: #1a1a2e;
        margin-bottom: 4px;
    }

    .section-sub {
        font-size: 13px;
        color: #aaa;
        margin-bottom: 24px;
    }

    .result-card-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        border-radius: 16px;
        padding: 28px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(238,90,36,0.25);
    }

    .result-card-low {
        background: linear-gradient(135deg, #26de81, #20bf6b);
        border-radius: 16px;
        padding: 28px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(32,191,107,0.25);
    }

    .result-model {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.75;
        margin-bottom: 10px;
    }

    .result-icon { font-size: 32px; margin: 8px 0; }
    .result-text { font-size: 18px; font-weight: 600; }

    .consensus-high {
        background: #fff5f5;
        border: 1.5px solid #ffb8b8;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .consensus-low {
        background: #f0fff8;
        border: 1.5px solid #b8f5d4;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .consensus-title {
        font-family: 'DM Serif Display', serif;
        font-size: 24px;
        margin: 8px 0 4px;
    }

    /* Sidebar nav buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #aaa !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        border-radius: 0 !important;
        padding: 12px 24px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-align: left !important;
        box-shadow: none !important;
        width: 100% !important;
        justify-content: flex-start !important;
        margin-bottom: 2px !important;
        transform: none !important;
        transition: all 0.2s !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        color: white !important;
        background: rgba(255,255,255,0.06) !important;
        border-left: 3px solid #e84393 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Main predict button */
    .main-btn > .stButton > button {
        background: linear-gradient(135deg, #e84393, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(232,67,147,0.3) !important;
    }

    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 1.5px solid #eeebe6 !important;
        background: #ffffff !important;
    }

    .input-section-title {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #e84393;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1.5px solid #fce4f0;
    }

    .divider {
        height: 1px;
        background: #f5f2ee;
        margin: 32px 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# --- Load Models ---
@st.cache_resource
def load_models():
    knn = joblib.load('models/knn_model.pkl')
    svm = joblib.load('models/svm_model.pkl')
    ann = joblib.load('models/ann_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return knn, svm, ann, scaler

@st.cache_data
def load_data():
    return pd.read_csv('data/raw/heart.csv')

knn, svm, ann, scaler = load_models()
df = load_data()

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🫀 CardioSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    for icon, label in [("🏠", "Home"), ("🔍", "Predict"), ("📊", "Model Performance"), ("📈", "Dataset Overview")]:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px; color:#444; text-align:center; padding:0 1rem;">TARUMT AI Assignment<br>Session 202601</p>', unsafe_allow_html=True)

page = st.session_state.page

# ==================== HOME ====================
if page == "Home":
    st.markdown('<div class="hero-title">Heart Disease<br>Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Three machine learning models working together to assess cardiovascular risk</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, label, val, sub in [
        (col1, "Dataset Size", "1,025", "Patient records"),
        (col2, "Features", "13", "Clinical attributes"),
        (col3, "Best Accuracy", "98.5%", "ANN Model"),
        (col4, "Missing Values", "Zero", "Clean dataset"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="section-title">About This Project</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">TARUMT Artificial Intelligence Assignment — Session 202601</div>', unsafe_allow_html=True)
        st.write("""
        This application predicts the likelihood of heart disease based on 13 clinical
        attributes using three supervised machine learning models. Each model was trained
        on the UCI Heart Disease dataset and evaluated using accuracy, precision, recall,
        and F1 score.
        """)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        for c, model, acc, color in [
            (col_a, "KNN", "83.4%", "#1a1a2e"),
            (col_b, "SVM", "88.8%", "#1a1a2e"),
            (col_c, "ANN", "98.5%", "#e84393"),
        ]:
            with c:
                st.markdown(f"""
                <div class="metric-card" style="text-align:center">
                    <div class="metric-label">{model}</div>
                    <div style="font-size:28px; font-family:'DM Serif Display'; color:{color}">{acc}</div>
                    <div class="metric-sub">Accuracy</div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Models Used</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Each member implemented one classifier</div>', unsafe_allow_html=True)
        for model, desc, acc in [
            ("K-Nearest Neighbors", "Instance-based, K=5, Euclidean distance.", "83.41%"),
            ("Support Vector Machine", "RBF kernel, C=1.0. Margin maximization.", "88.78%"),
            ("Artificial Neural Network", "MLP (32,16), ReLU, 500 iterations.", "98.54%"),
        ]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{acc} accuracy</div>
                <div style="font-size:15px; font-weight:600; color:#1a1a2e; margin:4px 0">{model}</div>
                <div style="font-size:12px; color:#aaa">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ==================== PREDICT ====================
elif page == "Predict":
    st.markdown('<div class="hero-title">Patient Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter clinical data to receive predictions from all three models</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="input-section-title">Demographics</div>', unsafe_allow_html=True)
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0,1,2,3],
                          format_func=lambda x: ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"][x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        st.markdown('<div class="input-section-title">Clinical Tests</div>', unsafe_allow_html=True)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1],
                           format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0,1,2],
                               format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0,1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0)

    with col3:
        st.markdown('<div class="input-section-title">Additional Features</div>', unsafe_allow_html=True)
        slope = st.selectbox("Slope of ST Segment", [0,1,2],
                             format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
        ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [0,1,2,3],
                            format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect","Other"][x])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("🫀  Analyse Patient Risk"):
        input_data = np.array([[
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]], dtype=float)

        input_scaled = scaler.transform(input_data)
        knn_pred = int(knn.predict(input_scaled)[0])
        svm_pred = int(svm.predict(input_scaled)[0])
        ann_pred = int(ann.predict(input_scaled)[0])

        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Individual model predictions</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        for col, model_name, pred in [(col1,"KNN",knn_pred),(col2,"SVM",svm_pred),(col3,"ANN",ann_pred)]:
            with col:
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-card-high">
                        <div class="result-model">{model_name}</div>
                        <div class="result-icon">⚠️</div>
                        <div class="result-text">High Risk</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-low">
                        <div class="result-model">{model_name}</div>
                        <div class="result-icon">✅</div>
                        <div class="result-text">Low Risk</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        votes = knn_pred + svm_pred + ann_pred
        if votes >= 2:
            st.markdown("""
            <div class="consensus-high">
                <div style="font-size:28px">⚠️</div>
                <div class="consensus-title" style="color:#c0392b">High Risk Consensus</div>
                <div style="font-size:13px; color:#aaa">2 or more models indicate elevated cardiovascular risk</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="consensus-low">
                <div style="font-size:28px">✅</div>
                <div class="consensus-title" style="color:#20bf6b">Low Risk Consensus</div>
                <div style="font-size:13px; color:#aaa">2 or more models indicate low cardiovascular risk</div>
            </div>
            """, unsafe_allow_html=True)

# ==================== MODEL PERFORMANCE ====================
elif page == "Model Performance":
    st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Evaluation metrics comparison across all three classifiers</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, model, acc, prec, rec, f1, color in [
        (col1, "KNN", "83.41%", "80.00%", "89.32%", "84.40%", "#4C72B0"),
        (col2, "SVM", "88.78%", "85.09%", "94.17%", "89.40%", "#DD8452"),
        (col3, "ANN", "98.54%", "100.00%", "97.09%", "98.52%", "#e84393"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px">
                    <div class="metric-label">{model} Classifier</div>
                    <div style="width:10px; height:10px; border-radius:50%; background:{color}"></div>
                </div>
                <div style="font-family:'DM Serif Display',serif; font-size:36px; color:{color}; margin:8px 0">{acc}</div>
                <div style="height:1px; background:#f5f2ee; margin:12px 0"></div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
                    <div><div class="metric-label">Precision</div><div style="font-size:16px; font-weight:600; color:#1a1a2e; margin-top:2px">{prec}</div></div>
                    <div><div class="metric-label">Recall</div><div style="font-size:16px; font-weight:600; color:#1a1a2e; margin-top:2px">{rec}</div></div>
                    <div style="grid-column:span 2"><div class="metric-label">F1 Score</div><div style="font-size:16px; font-weight:600; color:#1a1a2e; margin-top:2px">{f1}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="section-title">Performance Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">All metrics across models</div>', unsafe_allow_html=True)
        results = pd.DataFrame({
            'Model': ['KNN', 'SVM', 'ANN'],
            'Accuracy': [83.41, 88.78, 98.54],
            'Precision': [80.00, 85.09, 100.00],
            'Recall': [89.32, 94.17, 97.09],
            'F1 Score': [84.40, 89.40, 98.52]
        })
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        x = np.arange(3)
        width = 0.2
        for i, (metric, color) in enumerate(zip(
            ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            ['#4C72B0', '#DD8452', '#55A868', '#e84393']
        )):
            bars = ax.bar(x + (i-1.5)*width, results[metric], width,
                         label=metric, color=color, alpha=0.85, zorder=3)
            for b in bars:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                       f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7, color='#555')
        ax.set_xticks(x)
        ax.set_xticklabels(['KNN', 'SVM', 'ANN'], fontsize=12)
        ax.set_ylim(60, 110)
        ax.set_ylabel('Score (%)', color='#aaa', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.2, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#eeebe6')
        ax.spines['bottom'].set_color('#eeebe6')
        ax.tick_params(colors='#aaa')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Prediction breakdown per model</div>', unsafe_allow_html=True)
        for model_name, cm in [
            ('KNN', np.array([[79, 23], [11, 92]])),
            ('SVM', np.array([[85, 17], [6, 97]])),
            ('ANN', np.array([[102, 0], [3, 100]])),
        ]:
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            sns.heatmap(cm, annot=True, fmt='d',
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'],
                       cmap='Blues', ax=ax, cbar=False,
                       linewidths=0.5, linecolor='#f5f2ee')
            ax.set_title(model_name, fontsize=11, fontweight='600', color='#1a1a2e')
            ax.set_xlabel('Predicted', fontsize=9, color='#aaa')
            ax.set_ylabel('Actual', fontsize=9, color='#aaa')
            ax.tick_params(colors='#aaa', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)

# ==================== DATASET OVERVIEW ====================
elif page == "Dataset Overview":
    st.markdown('<div class="hero-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">UCI Heart Disease Dataset — 1025 patient records, 13 clinical features</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, label, val, sub in [
        (col1, "Total Samples", "1,025", "Patient records"),
        (col2, "Features", "13", "Clinical attributes"),
        (col3, "Missing Values", "0", "Clean dataset"),
        (col4, "Class Balance", "51/49", "Disease vs healthy"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:32px">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        counts = df['target'].value_counts()
        bars = ax.bar(['No Disease', 'Disease'], counts.values,
                     color=['#4C72B0', '#e84393'], alpha=0.85, width=0.5, zorder=3)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                   str(int(b.get_height())), ha='center', fontsize=12, fontweight='600', color='#1a1a2e')
        ax.set_ylabel('Count', color='#aaa')
        ax.grid(axis='y', alpha=0.2, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#eeebe6')
        ax.spines['bottom'].set_color('#eeebe6')
        ax.tick_params(colors='#aaa')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#ffffff')
        sns.heatmap(df.corr(), annot=False, cmap='RdBu_r', ax=ax,
                   linewidths=0.3, linecolor='#f5f2ee', cbar=True)
        ax.tick_params(colors='#aaa', labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Distribution of all 13 clinical features</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    fig.patch.set_facecolor('#ffffff')
    axes = axes.flatten()
    colors = ['#4C72B0','#DD8452','#55A868','#e84393','#9467bd',
              '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
              '#4C72B0','#DD8452','#55A868']
    for i, col in enumerate(df.columns[:-1]):
        axes[i].hist(df[col], bins=20, color=colors[i], alpha=0.8, edgecolor='white')
        axes[i].set_title(col, fontsize=10, fontweight='600', color='#1a1a2e')
        axes[i].set_facecolor('#ffffff')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_color('#eeebe6')
        axes[i].spines['bottom'].set_color('#eeebe6')
        axes[i].tick_params(colors='#aaa', labelsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
