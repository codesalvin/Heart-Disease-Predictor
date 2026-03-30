# ==================== SECTION 3: DATASET OVERVIEW ====================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="label-caps">Data Intelligence</p>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 3rem;">Dataset Overview</h2>', unsafe_allow_html=True)

if df is not None:
    # Basic Stats Metrics
    st.markdown('<div class="bento-card editorial-shadow">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Samples", len(df), delta="Verified")
    m2.metric("Clinical Features", len(df.columns)-1, delta="Structured")
    m3.metric("Missing Values", df.isnull().sum().sum(), delta="Clean", delta_color="normal")
    
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
