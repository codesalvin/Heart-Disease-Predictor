import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# --- Page Config ---
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (ported from HTML design system) ---
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,200..800;1,6..72,200..800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    /* ── Design Tokens ── */
    :root {
        --primary:                  #b10c69;
        --primary-container:        #d33182;
        --on-primary:               #ffffff;
        --on-primary-container:     #fffbff;
        --inverse-primary:          #ffb0cc;
        --primary-fixed:            #ffd9e4;
        --primary-fixed-dim:        #ffb0cc;

        --secondary:                #5d5c74;
        --secondary-container:      #e2e0fc;
        --on-secondary:             #ffffff;
        --on-secondary-container:   #63627a;

        --tertiary:                 #006a39;
        --tertiary-container:       #00864a;
        --on-tertiary:              #ffffff;
        --on-tertiary-container:    #f6fff4;
        --tertiary-fixed:           #5cff9f;
        --tertiary-fixed-dim:       #2ee285;

        --error:                    #ba1a1a;
        --error-container:          #ffdad6;
        --on-error:                 #ffffff;
        --on-error-container:       #93000a;

        --surface:                  #faf9f6;
        --surface-bright:           #faf9f6;
        --surface-dim:              #dbdad7;
        --surface-variant:          #e3e2df;
        --surface-tint:             #b5116b;
        --surface-container-lowest: #ffffff;
        --surface-container-low:    #f4f3f0;
        --surface-container:        #efeeeb;
        --surface-container-high:   #e9e8e5;
        --surface-container-highest:#e3e2df;

        --on-surface:               #1a1c1a;
        --on-surface-variant:       #584048;
        --inverse-surface:          #2f312f;
        --inverse-on-surface:       #f2f1ee;

        --outline:                  #8b7078;
        --outline-variant:          #debec8;

        --sidebar-bg:               #1a1a2e;

        --font-headline: 'Newsreader', serif;
        --font-body:     'Inter', sans-serif;

        --shadow-editorial: 0 32px 64px -12px rgba(26,28,26,0.06);
    }

    /* ── Base ── */
    html, body, [class*="css"], .stApp {
        font-family: var(--font-body) !important;
        background-color: var(--surface) !important;
        color: var(--on-surface) !important;
    }

    .main, .block-container {
        background-color: var(--surface) !important;
        padding-top: 2.5rem !important;
        padding-bottom: 4rem !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: var(--sidebar-bg) !important;
        padding-top: 2rem !important;
    }
    /* ── Sidebar base ── */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: var(--sidebar-bg) !important;
        padding-top: 2rem !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        font-family: var(--font-body) !important;
    }
    /* ── Sidebar radio nav ── */
    /* hide the generated label above the radio group */
    section[data-testid="stSidebar"] .stRadio > div:first-child {
        display: none !important;
    }
    /* each radio option wrapper */
    section[data-testid="stSidebar"] .stRadio > div > div {
        gap: 0 !important;
    }
    /* the clickable label row */
    section[data-testid="stSidebar"] .stRadio label {
        display: flex !important;
        align-items: center !important;
        padding: 11px 16px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        color: #94a3b8 !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        transition: background 0.15s, color 0.15s !important;
        width: 100% !important;
        margin: 1px 0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
        background: rgba(255,255,255,0.05) !important;
    }
    /* hide the actual circle dot */
    section[data-testid="stSidebar"] .stRadio label > div:first-child {
        display: none !important;
    }
    /* the text span */
    section[data-testid="stSidebar"] .stRadio label > div:last-child p,
    section[data-testid="stSidebar"] .stRadio label > div:last-child {
        color: inherit !important;
        font-size: 14px !important;
        font-weight: inherit !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    /* selected state — input:checked sibling label */
    section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] + label,
    section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div label,
    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        color: #ffffff !important;
        background: rgba(177,12,105,0.15) !important;
        border-right: 3px solid #b10c69 !important;
        font-weight: 500 !important;
    }
    /* sidebar hr */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 16px 0 !important;
    }

    /* Hide sidebar collapse button */
button[data-testid="collapsedControl"] {
    display: none !important;
}

/* Prevent sidebar from collapsing */
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
}

    /* ── Hero title ── */
    .cs-hero-title {
        font-family: var(--font-headline);
        font-size: clamp(2.5rem, 5vw, 3.75rem);
        color: var(--on-surface);
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin-bottom: 1rem;
    }
    .cs-hero-title .accent {
        color: var(--primary);
        font-style: italic;
    }
    .cs-hero-sub {
        font-size: 1.1rem;
        color: var(--on-surface-variant);
        font-weight: 300;
        line-height: 1.65;
        max-width: 42ch;
        margin-bottom: 2rem;
    }

    /* ── Section labels ── */
    .cs-eyebrow {
        font-size: 0.6875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--primary);
        margin-bottom: 0.375rem;
    }
    .cs-section-title {
        font-family: var(--font-headline);
        font-size: 1.75rem;
        color: var(--on-surface);
        margin-bottom: 0.25rem;
    }
    .cs-section-sub {
        font-size: 0.8125rem;
        color: var(--on-surface-variant);
        margin-bottom: 1.5rem;
    }

    /* ── Metric / Bento card ── */
    .cs-metric-card {
        background: var(--surface-container-lowest);
        border-radius: 12px;
        padding: 28px;
        border: 1px solid rgba(222,190,200,0.12);
        box-shadow: var(--shadow-editorial);
        margin-bottom: 0;
    }
    .cs-metric-label {
        font-size: 0.6875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--on-surface-variant);
        margin-bottom: 14px;
    }
    .cs-metric-value {
        font-family: var(--font-headline);
        font-size: 2.75rem;
        color: var(--on-surface);
        line-height: 1;
        margin-bottom: 4px;
    }
    .cs-metric-sub {
        font-size: 0.75rem;
        color: var(--on-surface-variant);
    }

    /* ── Model card ── */
    .cs-model-card {
        background: var(--surface-container-lowest);
        border-radius: 12px;
        padding: 24px 28px;
        border: 1px solid rgba(222,190,200,0.10);
        box-shadow: var(--shadow-editorial);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        transition: transform 0.18s ease;
    }
    .cs-model-card:hover { transform: scale(1.008); }
    .cs-model-card .icon-wrap {
        width: 52px; height: 52px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin-right: 18px;
        flex-shrink: 0;
    }
    .cs-model-card .title { font-family: var(--font-headline); font-size: 1.2rem; }
    .cs-model-card .subtitle { font-size: 0.75rem; color: var(--on-surface-variant); }
    .cs-model-card .acc { font-family: var(--font-headline); font-size: 2rem; line-height:1; }
    .cs-model-card .acc-label { font-size: 0.6rem; text-transform:uppercase; letter-spacing:.12em; color: var(--on-surface-variant); }

    /* ── Progress bars (telemetry) ── */
    .cs-telemetry {
        background: var(--surface-container-low);
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
    }
    .cs-telemetry-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 20px;
    }
    .cs-telemetry-label {
        font-size: 0.6875rem; font-weight:700; text-transform:uppercase; letter-spacing:.18em;
    }
    .cs-live {
        font-size: 0.6875rem; color: var(--tertiary);
    }
    .cs-bar-row { display:flex; align-items:center; gap:14px; margin-bottom:14px; }
    .cs-bar-track {
        flex:1; height:6px; background: var(--surface-container-highest);
        border-radius: 9999px; overflow:hidden;
    }
    .cs-bar-fill { height:100%; border-radius:9999px; }
    .cs-bar-val { font-size: 0.6rem; font-weight:600; color: var(--on-surface-variant); width:28px; }

    /* ── Input section title ── */
    .cs-input-eyebrow {
        font-size: 0.6875rem; font-weight:700; text-transform:uppercase; letter-spacing:.18em;
        color: var(--primary); padding-bottom:8px;
        border-bottom: 1.5px solid rgba(177,12,105,0.15);
        margin-bottom:14px;
    }

    /* ── Result cards ── */
    .cs-result-high {
        background: linear-gradient(135deg,#ff6b6b,#ee5a24);
        border-radius:14px; padding:24px; color:white; text-align:center;
        box-shadow: 0 8px 28px rgba(238,90,36,0.28);
    }
    .cs-result-low {
        background: linear-gradient(135deg,#26de81,#20bf6b);
        border-radius:14px; padding:24px; color:white; text-align:center;
        box-shadow: 0 8px 28px rgba(32,191,107,0.25);
    }
    .cs-result-model { font-size:10px; text-transform:uppercase; letter-spacing:2px; opacity:.8; margin-bottom:6px; }
    .cs-result-icon  { font-size:32px; margin:6px 0; }
    .cs-result-text  { font-size:17px; font-weight:600; }

    /* ── Consensus box ── */
    .cs-consensus-high {
        background: #fff5f5; border:2px solid #ff6b6b; border-radius:14px;
        padding:24px; text-align:center; margin-top:20px;
    }
    .cs-consensus-low {
        background: #f0fff4; border:2px solid #26de81; border-radius:14px;
        padding:24px; text-align:center; margin-top:20px;
    }
    .cs-consensus-icon  { font-size:30px; }
    .cs-consensus-title {
        font-family: var(--font-headline); font-size:1.375rem;
        margin: 8px 0 4px;
    }
    .cs-consensus-sub { font-size:13px; color: var(--on-surface-variant); }

    /* ── Perf metric card ── */
    .cs-perf-card {
        background: var(--surface-container-lowest);
        border-radius:12px; padding:22px 24px;
        border: 1px solid rgba(222,190,200,0.10);
        box-shadow: var(--shadow-editorial);
        margin-bottom:0;
    }
    .cs-perf-acc {
        font-family: var(--font-headline); font-size:2.25rem; line-height:1; margin:10px 0 14px;
    }
    .cs-perf-grid {
        display:grid; grid-template-columns:1fr 1fr; gap:10px;
        border-top:1px solid var(--surface-container-high); padding-top:12px; margin-top:4px;
    }
    .cs-perf-stat .label { font-size:0.625rem; text-transform:uppercase; letter-spacing:.14em; color: var(--on-surface-variant); }
    .cs-perf-stat .val   { font-size:1rem; font-weight:600; color: var(--on-surface); margin-top:2px; }

    /* ── About card ── */
    .cs-about-card {
        background: var(--surface-container-lowest);
        border-radius:12px; padding:28px 30px;
        border: 1px solid rgba(222,190,200,0.10);
        box-shadow: var(--shadow-editorial);
    }
    .cs-uni-badge {
        display:flex; align-items:center; gap:14px;
        border-top: 1px solid rgba(222,190,200,0.18); padding-top:18px; margin-top:20px;
    }
    .cs-uni-icon {
        width:44px; height:44px; border-radius:8px;
        background: var(--surface-container);
        display:flex; align-items:center; justify-content:center;
        font-size:20px; flex-shrink:0;
    }
    .cs-uni-name { font-size:0.8125rem; font-weight:700; color: var(--on-surface); }
    .cs-uni-sub  { font-size:0.625rem; text-transform:uppercase; letter-spacing:.1em; color: var(--on-surface-variant); }

    /* ── Divider ── */
    .cs-divider { height:1px; background: rgba(222,190,200,0.18); margin:2rem 0; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-container)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        font-family: var(--font-body) !important;
        width: 100% !important;
        box-shadow: 0 4px 18px rgba(177,12,105,0.25) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(177,12,105,0.35) !important;
    }

    /* ── Inputs ── */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 1.5px solid rgba(222,190,200,0.4) !important;
        font-family: var(--font-body) !important;
        background: var(--surface-container-lowest) !important;
    }
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(177,12,105,0.10) !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

    /* ── Hide streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    knn    = joblib.load('models/knn_model.pkl')
    svm    = joblib.load('models/svm_model.pkl')
    ann    = joblib.load('models/ann_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return knn, svm, ann, scaler

@st.cache_data
def load_data():
    return pd.read_csv('data/raw/heart.csv')

knn, svm, ann, scaler = load_models()
df = load_data()

# --- Session state nav ---
NAV_ITEMS = [
    ("🏠", "Home",             "Home"),
    ("🔍", "Predict",          "Predict"),
    ("📊", "Model Performance","Model Performance"),
    ("📈", "Dataset Overview", "Dataset Overview"),
]
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Sidebar: pure st.radio, hidden label, styled via stable p tag selectors ---
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 4px 4px 4px">
        <div style="font-family:'Newsreader',serif;font-size:26px;color:#fff;letter-spacing:-0.01em">CardioSense AI</div>
        <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.22em;color:#b10c69;margin-top:2px">Clinical Precision</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.08);margin:16px 0">
    """, unsafe_allow_html=True)

    # st.radio is the only reliable Streamlit nav — we style the visible <p> elements
    page_labels = [f"{icon}  {label}" for icon, label, _ in NAV_ITEMS]
    page_keys   = {f"{icon}  {label}": key for icon, label, key in NAV_ITEMS}

    selected_label = st.radio(
        "nav",
        page_labels,
        index=[k for _,_,k in NAV_ITEMS].index(st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = page_keys[selected_label]

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.08);margin:16px 0">
    <p style="font-size:11px;color:#475569;text-align:center;padding:4px 0 0">
        TARUMT AI Assignment<br>Session 202601
    </p>""", unsafe_allow_html=True)

page = st.session_state.page

# ── helper ──────────────────────────────────────────────────────
def metric_card(label, value, sub="", accent_color=None):
    val_color = accent_color or "var(--on-surface)"
    return f"""
    <div class="cs-metric-card">
        <div class="cs-metric-label">{label}</div>
        <div class="cs-metric-value" style="color:{val_color}">{value}</div>
        <div class="cs-metric-sub">{sub}</div>
    </div>"""

# ==================== HOME ====================
if page == "Home":

    st.markdown("""
    <div class="cs-hero-title">
        Heart Disease<br>
        <span class="accent">Prediction System</span>
    </div>
    <div class="cs-hero-sub">
        An advanced clinical intelligence suite utilising three distinct machine learning
        models to synthesise cardiovascular risk factors into actionable diagnostic insights.
    </div>
    """, unsafe_allow_html=True)

    # Metric bento row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Dataset Size",    "1,025", "Records"),           unsafe_allow_html=True)
    c2.markdown(metric_card("Clinical Features","13",   "Parameters"),        unsafe_allow_html=True)
    c3.markdown(metric_card("Best Accuracy",   "98.5%", "ANN Model",
                            accent_color="var(--primary)"),                   unsafe_allow_html=True)
    c4.markdown(metric_card("Missing Values",  "Zero",  "Clean dataset",
                            accent_color="var(--tertiary)"),                  unsafe_allow_html=True)

    st.markdown('<div class="cs-divider"></div>', unsafe_allow_html=True)

    col_about, col_models = st.columns([5, 7])

    with col_about:
        st.markdown("""
        <div class="cs-about-card">
            <div class="cs-eyebrow">Foundation</div>
            <div class="cs-section-title" style="margin-bottom:16px">About This Project</div>
            <p style="color:var(--on-surface-variant);line-height:1.7;font-size:14px;margin-bottom:12px">
                Developed as part of the <strong style="color:var(--on-surface)">TARUMT AI Assignment</strong>,
                this project explores the intersection of supervised machine learning and clinical diagnostics.
            </p>
            <p style="color:var(--on-surface-variant);line-height:1.7;font-size:14px">
                The core objective is to evaluate how different algorithmic architectures—ranging from
                instance-based learning (KNN) to complex neural structures (ANN)—interpret the
                physiological signals that precede major cardiac events.
            </p>
            <div class="cs-uni-badge">
                <div class="cs-uni-icon">🎓</div>
                <div>
                    <div class="cs-uni-name">TARUMT University</div>
                    <div class="cs-uni-sub">Artificial Intelligence Faculty</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_models:
        st.markdown('<div class="cs-eyebrow" style="color:var(--on-surface-variant)">Model Benchmarking</div>', unsafe_allow_html=True)

        models = [
            ("psychology", "var(--primary)", "rgba(177,12,105,0.10)",
             "Artificial Neural Network (ANN)", "Multi-layer perceptron architecture",
             "98.5%", "var(--primary)"),
            ("border_inner", "var(--secondary)", "rgba(93,92,116,0.10)",
             "Support Vector Machine (SVM)", "Linear and RBF kernel optimisation",
             "88.8%", "var(--on-surface)"),
            ("hub", "var(--on-surface-variant)", "var(--surface-container-high)",
             "K-Nearest Neighbors (KNN)", "Instance-based classification (k=7)",
             "83.4%", "var(--on-surface-variant)"),
        ]

        for icon, icon_color, icon_bg, title, subtitle, acc, acc_color in models:
            st.markdown(f"""
            <div class="cs-model-card">
                <div style="display:flex;align-items:center">
                    <div class="icon-wrap" style="background:{icon_bg}">
                        <span class="material-symbols-outlined" style="color:{icon_color};font-size:26px">{icon}</span>
                    </div>
                    <div>
                        <div class="title">{title}</div>
                        <div class="subtitle">{subtitle}</div>
                    </div>
                </div>
                <div style="text-align:right;flex-shrink:0">
                    <div class="acc" style="color:{acc_color}">{acc}</div>
                    <div class="acc-label">Accuracy Rate</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Telemetry bars
        st.markdown("""
        <div class="cs-telemetry">
            <div class="cs-telemetry-header">
                <span class="cs-telemetry-label">Real-time Model Load</span>
                <span class="cs-live">● System Operational</span>
            </div>
            <div class="cs-bar-row">
                <div class="cs-bar-track"><div class="cs-bar-fill" style="width:98.5%;background:var(--primary)"></div></div>
                <span class="cs-bar-val">98.5</span>
            </div>
            <div class="cs-bar-row">
                <div class="cs-bar-track"><div class="cs-bar-fill" style="width:88.8%;background:var(--secondary)"></div></div>
                <span class="cs-bar-val">88.8</span>
            </div>
            <div class="cs-bar-row" style="margin-bottom:0">
                <div class="cs-bar-track"><div class="cs-bar-fill" style="width:83.4%;background:rgba(88,64,72,0.25)"></div></div>
                <span class="cs-bar-val">83.4</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== PREDICT ====================
elif page == "Predict":
    st.markdown("""
    <div class="cs-hero-title">Patient Risk Assessment</div>
    <div class="cs-hero-sub">Enter clinical data to receive predictions from all three models.</div>
    <div class="cs-divider"></div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="cs-input-eyebrow">Demographics</div>', unsafe_allow_html=True)
        age      = st.number_input("Age", 20, 100, 50)
        sex      = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        cp       = st.selectbox("Chest Pain Type", [0,1,2,3],
                                format_func=lambda x: ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"][x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        st.markdown('<div class="cs-input-eyebrow">Clinical Tests</div>', unsafe_allow_html=True)
        fbs     = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0,1,2],
                               format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang   = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0)

    with col3:
        st.markdown('<div class="cs-input-eyebrow">Additional Features</div>', unsafe_allow_html=True)
        slope = st.selectbox("Slope of ST Segment", [0,1,2],
                             format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
        ca    = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
        thal  = st.selectbox("Thalassemia", [0,1,2,3],
                             format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect","Other"][x])

    st.markdown('<div class="cs-divider"></div>', unsafe_allow_html=True)

    if st.button("🫀  Analyse Patient Risk"):
        input_data   = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]], dtype=float)
        input_scaled = scaler.transform(input_data)

        knn_pred = int(knn.predict(input_scaled)[0])
        svm_pred = int(svm.predict(input_scaled)[0])
        ann_pred = int(ann.predict(input_scaled)[0])

        st.markdown('<div class="cs-section-title">Prediction Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="cs-section-sub">Individual model predictions</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)

        def result_card(col, name, pred):
            css = "cs-result-high" if pred else "cs-result-low"
            icon = "⚠️" if pred else "✅"
            text = "High Risk" if pred else "Low Risk"
            with col:
                st.markdown(f"""
                <div class="{css}">
                    <div class="cs-result-model">{name}</div>
                    <div class="cs-result-icon">{icon}</div>
                    <div class="cs-result-text">{text}</div>
                </div>""", unsafe_allow_html=True)

        result_card(r1, "K-Nearest Neighbors", knn_pred)
        result_card(r2, "Support Vector Machine", svm_pred)
        result_card(r3, "Artificial Neural Network", ann_pred)

        votes = knn_pred + svm_pred + ann_pred
        if votes >= 2:
            st.markdown("""
            <div class="cs-consensus-high">
                <div class="cs-consensus-icon">⚠️</div>
                <div class="cs-consensus-title" style="color:#c0392b">High Risk Consensus</div>
                <div class="cs-consensus-sub">2 or more models indicate elevated cardiovascular risk.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="cs-consensus-low">
                <div class="cs-consensus-icon">✅</div>
                <div class="cs-consensus-title" style="color:#20bf6b">Low Risk Consensus</div>
                <div class="cs-consensus-sub">2 or more models indicate low cardiovascular risk.</div>
            </div>""", unsafe_allow_html=True)

# ==================== MODEL PERFORMANCE ====================
elif page == "Model Performance":
    st.markdown("""
    <div class="cs-hero-title">Model Performance</div>
    <div class="cs-hero-sub">Evaluation metrics comparison across all three classifiers.</div>
    <div class="cs-divider"></div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    perf = [
        (c1, "KNN", "83.41%", "80.00%", "89.32%", "84.40%", "var(--secondary)"),
        (c2, "SVM", "88.78%", "85.09%", "94.17%", "89.40%", "var(--on-surface)"),
        (c3, "ANN", "98.54%", "100.00%","97.09%", "98.52%", "var(--primary)"),
    ]
    for col, model, acc, prec, rec, f1, color in perf:
        with col:
            st.markdown(f"""
            <div class="cs-perf-card">
                <div class="cs-metric-label">{model} Classifier</div>
                <div class="cs-perf-acc" style="color:{color}">{acc}</div>
                <div class="cs-perf-grid">
                    <div class="cs-perf-stat"><div class="label">Precision</div><div class="val">{prec}</div></div>
                    <div class="cs-perf-stat"><div class="label">Recall</div><div class="val">{rec}</div></div>
                    <div class="cs-perf-stat" style="grid-column:span 2"><div class="label">F1 Score</div><div class="val">{f1}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cs-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="cs-section-title">Performance Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="cs-section-sub">All metrics across models</div>', unsafe_allow_html=True)

        results = pd.DataFrame({
            'Model':    ['KNN','SVM','ANN'],
            'Accuracy': [83.41,88.78,98.54],
            'Precision':[80.00,85.09,100.00],
            'Recall':   [89.32,94.17,97.09],
            'F1 Score': [84.40,89.40,98.52]
        })
        fig, ax = plt.subplots(figsize=(9,4))
        fig.patch.set_facecolor('#faf9f6')
        ax.set_facecolor('#faf9f6')
        x      = np.arange(3)
        width  = 0.2
        colors  = ['#b10c69','#5d5c74','#006a39','#debec8']
        metrics = ['Accuracy','Precision','Recall','F1 Score']
        for i,(metric,c) in enumerate(zip(metrics,colors)):
            bars = ax.bar(x+(i-1.5)*width, results[metric], width, label=metric, color=c, alpha=0.85, zorder=3)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                        f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7, color='#584048')
        ax.set_xticks(x); ax.set_xticklabels(['KNN','SVM','ANN'],fontsize=12)
        ax.set_ylim(60,110); ax.set_ylabel('Score (%)',color='#8b7078',fontsize=10)
        ax.legend(loc='lower right',fontsize=9)
        ax.grid(axis='y',alpha=0.25,zorder=0)
        for spine in ['top','right']: ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#debec8'); ax.spines['bottom'].set_color('#debec8')
        ax.tick_params(colors='#8b7078')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="cs-section-title">Confusion Matrices</div>', unsafe_allow_html=True)
        st.markdown('<div class="cs-section-sub">Prediction breakdown per model</div>', unsafe_allow_html=True)

        cm_data = {
            'KNN': (np.array([[79,23],[11,92]]),  '#5d5c74'),
            'SVM': (np.array([[85,17],[6, 97]]),  '#584048'),
            'ANN': (np.array([[102,0],[3,100]]),  '#b10c69'),
        }
        for model_name,(cm,color) in cm_data.items():
            fig,ax = plt.subplots(figsize=(3.5,2.5))
            fig.patch.set_facecolor('#f4f3f0')
            ax.set_facecolor('#f4f3f0')
            sns.heatmap(cm, annot=True, fmt='d',
                        xticklabels=['No Disease','Disease'],
                        yticklabels=['No Disease','Disease'],
                        cmap='RdPu', ax=ax, linewidths=0.5,
                        linecolor='#efeeeb', cbar=False)
            ax.set_title(model_name,fontsize=11,fontweight='bold',color='#1a1a2e')
            ax.set_xlabel('Predicted',fontsize=9,color='#8b7078')
            ax.set_ylabel('Actual',fontsize=9,color='#8b7078')
            ax.tick_params(colors='#8b7078',labelsize=8)
            plt.tight_layout(); st.pyplot(fig)

# ==================== DATASET OVERVIEW ====================
elif page == "Dataset Overview":
    st.markdown("""
    <div class="cs-hero-title">Dataset Overview</div>
    <div class="cs-hero-sub">UCI Heart Disease Dataset — 1,025 patient records, 13 clinical features.</div>
    <div class="cs-divider"></div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(metric_card("Total Samples",  "1,025", "Patient records"),  unsafe_allow_html=True)
    c2.markdown(metric_card("Features",       "13",    "Clinical attributes"),unsafe_allow_html=True)
    c3.markdown(metric_card("Missing Values", "0",     "Clean dataset",
                            accent_color="var(--tertiary)"),                  unsafe_allow_html=True)
    c4.markdown(metric_card("Class Balance",  "51/49", "Disease vs healthy"), unsafe_allow_html=True)

    st.markdown('<div class="cs-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    BG = '#faf9f6'

    with col1:
        st.markdown('<div class="cs-section-title">Class Distribution</div>', unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(5,3.5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        counts = df['target'].value_counts()
        bars   = ax.bar(['No Disease','Disease'], counts.values,
                        color=['#5d5c74','#b10c69'], alpha=0.85, width=0.5, zorder=3)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+5,
                    str(int(b.get_height())), ha='center', fontsize=12, fontweight='600', color='#1a1a2e')
        ax.set_ylabel('Count',color='#8b7078')
        ax.grid(axis='y',alpha=0.25,zorder=0)
        for s in ['top','right']: ax.spines[s].set_visible(False)
        ax.spines['left'].set_color('#debec8'); ax.spines['bottom'].set_color('#debec8')
        ax.tick_params(colors='#8b7078')
        plt.tight_layout(); st.pyplot(fig)

    with col2:
        st.markdown('<div class="cs-section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(5,3.5))
        fig.patch.set_facecolor(BG)
        sns.heatmap(df.corr(), annot=False, cmap='RdPu', ax=ax,
                    linewidths=0.3, linecolor='#efeeeb', cbar=True)
        ax.tick_params(colors='#8b7078', labelsize=7)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="cs-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-section-title">Feature Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-section-sub">Distribution of all 13 clinical features</div>', unsafe_allow_html=True)

    palette = ['#b10c69','#5d5c74','#006a39','#d33182','#8b7078',
               '#debec8','#b10c69','#5d5c74','#006a39','#d33182',
               '#b10c69','#5d5c74','#006a39']
    fig, axes = plt.subplots(3,5,figsize=(15,8))
    fig.patch.set_facecolor(BG); axes = axes.flatten()
    for i,col in enumerate(df.columns[:-1]):
        axes[i].hist(df[col], bins=20, color=palette[i], alpha=0.82, edgecolor='white')
        axes[i].set_title(col, fontsize=10, fontweight='600', color='#1a1a2e')
        axes[i].set_facecolor(BG)
        for s in ['top','right']: axes[i].spines[s].set_visible(False)
        axes[i].spines['left'].set_color('#debec8')
        axes[i].spines['bottom'].set_color('#debec8')
        axes[i].tick_params(colors='#8b7078',labelsize=7)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="cs-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-section-title">Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
