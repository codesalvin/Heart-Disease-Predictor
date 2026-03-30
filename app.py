import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- Page Config ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# --- Load Models ---
knn = joblib.load('models/knn_model.pkl')
svm = joblib.load('models/svm_model.pkl')
ann = joblib.load('models/ann_model.pkl')
scaler = joblib.load('models/scaler.pkl')
df = pd.read_csv('data/raw/heart.csv')

# --- Sidebar Navigation ---
st.sidebar.title("❤️ Heart Disease Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔍 Predict",
    "📊 Model Performance",
    "📈 Dataset Overview"
])

# ==================== HOME ====================
if page == "🏠 Home":
    st.title("Heart Disease Prediction System")
    st.markdown("### Using Machine Learning to Predict Cardiovascular Risk")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Dataset**\n\n1025 patient records\n\n13 clinical features")
    with col2:
        st.info("**Models**\n\nKNN | SVM | ANN")
    with col3:
        st.info("**Best Accuracy**\n\nANN: 98.54%")

    st.markdown("---")
    st.markdown("### About This Project")
    st.write("""
        This application was developed as part of an Artificial Intelligence assignment
        at TARUMT. It predicts the likelihood of heart disease in a patient based on
        13 clinical attributes using three machine learning models: K-Nearest Neighbors (KNN),
        Support Vector Machine (SVM), and Artificial Neural Network (ANN).
    """)

    st.markdown("### Model Performance Summary")
    results = pd.DataFrame({
        'Model': ['KNN', 'SVM', 'ANN'],
        'Accuracy': ['83.41%', '88.78%', '98.54%'],
        'Precision': ['80.00%', '85.09%', '100.00%'],
        'Recall': ['89.32%', '94.17%', '97.09%'],
        'F1 Score': ['84.40%', '89.40%', '98.52%']
    })
    st.dataframe(results, use_container_width=True)

# ==================== PREDICT ====================
elif page == "🔍 Predict":
    st.title("🔍 Heart Disease Prediction")
    st.markdown("Enter patient details below to get predictions from all 3 models.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0,1,2,3],
                          format_func=lambda x: ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"][x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

    with col2:
        st.markdown("**Clinical Tests**")
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1],
                           format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0,1,2],
                               format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0,1],
                             format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0)

    with col3:
        st.markdown("**Additional Features**")
        slope = st.selectbox("Slope of ST Segment", [0,1,2],
                             format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
        ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [0,1,2,3],
                            format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect","Other"][x])

    st.markdown("---")

    # --- ALL variables are now defined before this button ---
    if st.button("🔍 Predict with All Models", use_container_width=True):

        input_data = np.array([[
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]], dtype=float)

        input_scaled = scaler.transform(input_data)

        knn_pred = int(knn.predict(input_scaled)[0])
        svm_pred = int(svm.predict(input_scaled)[0])
        ann_pred = int(ann.predict(input_scaled)[0])

        st.markdown("### Prediction Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            if knn_pred == 1:
                st.error("**KNN**\n\n⚠️ High Risk")
            else:
                st.success("**KNN**\n\n✅ Low Risk")

        with col2:
            if svm_pred == 1:
                st.error("**SVM**\n\n⚠️ High Risk")
            else:
                st.success("**SVM**\n\n✅ Low Risk")

        with col3:
            if ann_pred == 1:
                st.error("**ANN**\n\n⚠️ High Risk")
            else:
                st.success("**ANN**\n\n✅ Low Risk")

        st.markdown("---")
        votes = knn_pred + svm_pred + ann_pred
        if votes >= 2:
            st.error("### ⚠️ Consensus: HIGH risk of heart disease (2 or more models agree)")
        else:
            st.success("### ✅ Consensus: LOW risk of heart disease (2 or more models agree)")

# ==================== MODEL PERFORMANCE ====================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("---")

    # Metrics Table
    st.markdown("### Evaluation Metrics Comparison")
    results = pd.DataFrame({
        'Model': ['KNN', 'SVM', 'ANN'],
        'Accuracy': [83.41, 88.78, 98.54],
        'Precision': [80.00, 85.09, 100.00],
        'Recall': [89.32, 94.17, 97.09],
        'F1 Score': [84.40, 89.40, 98.52]
    })
    st.dataframe(results, use_container_width=True)

    # Bar Chart
    st.markdown("### Performance Comparison Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    width = 0.2
    bars1 = ax.bar(x - width*1.5, results['Accuracy'], width, label='Accuracy', color='#4C72B0')
    bars2 = ax.bar(x - width*0.5, results['Precision'], width, label='Precision', color='#DD8452')
    bars3 = ax.bar(x + width*0.5, results['Recall'], width, label='Recall', color='#55A868')
    bars4 = ax.bar(x + width*1.5, results['F1 Score'], width, label='F1 Score', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(['KNN', 'SVM', 'ANN'])
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    ax.set_ylim(0, 110)
    for bar in [bars1, bars2, bars3, bars4]:
        for b in bar:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                   f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7)
    st.pyplot(fig)

    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    cm_data = {
        'KNN': np.array([[79, 23], [11, 92]]),
        'SVM': np.array([[85, 17], [6, 97]]),
        'ANN': np.array([[102, 0], [3, 100]])
    }

    col1, col2, col3 = st.columns(3)
    for col, (model_name, cm) in zip([col1, col2, col3], cm_data.items()):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d',
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'],
                       cmap='Blues', ax=ax)
            ax.set_title(f'{model_name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

# ==================== DATASET OVERVIEW ====================
elif page == "📈 Dataset Overview":
    st.title("📈 Dataset Overview")
    st.markdown("---")

    # Basic Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", "1025")
    col2.metric("Features", "13")
    col3.metric("Missing Values", "0")

    st.markdown("---")

    # Sample data
    st.markdown("### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    # Class Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        df['target'].value_counts().plot(kind='bar', ax=ax,
                                          color=['#4C72B0', '#DD8452'])
        ax.set_xticklabels(['No Disease', 'Disease'], rotation=0)
        ax.set_title('Class Distribution')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    with col2:
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
        ax.set_title('Feature Correlation')
        st.pyplot(fig)

    # Feature Distributions
    st.markdown("### Feature Distributions")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    for i, col in enumerate(df.columns[:-1]):
        axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='black')
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
