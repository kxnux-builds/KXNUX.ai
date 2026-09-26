import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'salary_prediction_pipeline.joblib')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Salary_Data.csv')

st.set_page_config(
    page_title='Salary Predictor',
    page_icon='KXNUX.ai',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('''
<style>
.main { background: #f6f8fc; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827 0%, #1e3a8a 55%, #2563eb 100%);
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 18px 45px rgba(30,64,175,.22);
}
.hero h1 { font-size: 2.55rem; margin: 0 0 .45rem 0; font-weight: 800; }
.hero p { font-size: 1.06rem; margin: 0; opacity: .88; }
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.25rem;
    box-shadow: 0 8px 25px rgba(15,23,42,.06);
}
.result {
    background: linear-gradient(135deg, #ecfeff 0%, #eff6ff 100%);
    border: 1px solid #bfdbfe;
    border-radius: 22px;
    padding: 1.55rem;
    box-shadow: 0 12px 30px rgba(37,99,235,.10);
}
.salary { font-size: 2.4rem; font-weight: 850; color: #1d4ed8; margin: .2rem 0 .6rem; }
.small { color: #64748b; font-size: .92rem; }
.badge {
  display:inline-block; padding:.34rem .68rem; border-radius:999px;
  background:#dbeafe; color:#1e40af; font-weight:700; font-size:.82rem;
}
.section-title { font-size: 1.35rem; font-weight: 800; margin-top: .4rem; }
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 2.5rem;
    line-height: 1.6;
}

.footer span {
    background: linear-gradient(90deg, #818cf8, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

</style>
''', unsafe_allow_html=True)

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

def normalize_column(df, candidates):
    normalized = {str(c).strip().lower().replace('_', ' '): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace('_', ' ')
        if key in normalized:
            return normalized[key]
    return None

def unique_values(df, candidates, fallback):
    col = normalize_column(df, candidates)
    if col is None:
        return fallback
    values = sorted(df[col].dropna().astype(str).str.strip().unique().tolist())
    return values if values else fallback

def experience_category(years):
    if years <= 2:
        return 'Fresher / Entry Level'
    if years <= 5:
        return 'Junior'
    if years <= 10:
        return 'Mid-Level'
    return 'Senior / Lead'

def recommendation(years, predicted_salary, education):
    category = experience_category(years)
    if years <= 2:
        return f"Consider an entry-level package with structured onboarding and mentorship. Suggested band: {category}."
    if years <= 5:
        return f"Consider a competitive junior package and evaluate project depth, interview performance, and skill match."
    if years <= 10:
        return f"Consider a market-aligned mid-level package with role-based adjustments for technical ownership and leadership."
    return f"Consider a senior/lead package with role scope, leadership responsibility, and critical-skill premium considered."

st.markdown('''
<div class="hero">
  <h1>Salary Prediction System</h1>
  <p>AI-assisted salary estimation for faster, more consistent and data-driven HR decisions.</p>
</div>
''', unsafe_allow_html=True)

model = load_model()
metadata = load_metadata()
df = load_dataset()

if model is None:
    st.error('Model not found. Run `python train_model.py` after placing your dataset at `data/Salary_Data.csv`.')
    st.stop()

# Sidebar inputs
st.sidebar.markdown('## 👤 Candidate Profile')
st.sidebar.caption('Enter candidate details to generate an estimated annual salary.')

ages = (18, 70)
age = st.sidebar.slider('Age', min_value=18, max_value=70, value=25, step=1)

gender_options = unique_values(df, ['Gender', 'Sex'], ['Male', 'Female']) if df is not None else ['Male', 'Female']
gender = st.sidebar.selectbox('Gender', gender_options)

education_options = unique_values(df, ['Education Level', 'Education', 'Degree'], ['High School', "Bachelor's", "Master's", 'PhD']) if df is not None else ['High School', "Bachelor's", "Master's", 'PhD']
education = st.sidebar.selectbox('Education Level', education_options)

years = st.sidebar.slider('Years of Experience', min_value=0.0, max_value=40.0, value=2.0, step=0.5)

job_options = unique_values(df, ['Job Title', 'Job', 'Designation', 'Role'], ['Software Engineer']) if df is not None else ['Software Engineer']
job_title = st.sidebar.selectbox('Job Title', job_options)

predict_clicked = st.sidebar.button('🚀 Predict Salary', use_container_width=True, type='primary')

st.markdown('<div class="section-title">Prediction Dashboard</div>', unsafe_allow_html=True)

if predict_clicked:
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Years of Experience': years,
        'Job Title': job_title
    }])

    prediction = float(model.predict(input_df)[0])
    category = experience_category(years)
    advice = recommendation(years, prediction, education)

    c1, c2 = st.columns([1.35, 1], gap='large')
    with c1:
        st.markdown(f'''
        <div class="result">
          <div class="small">Estimated Annual Salary (dataset units)</div>
          <div class="salary">{prediction:,.0f}</div>
          <span class="badge">{category}</span>
          <p class="small" style="margin-top:1rem;">Model estimate based on the historical training data and selected candidate profile. Salary is shown in the same units as the training dataset.</p>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('### HR Recommendation')
        st.write(advice)
        st.markdown(f'**Education:** {education}')
        st.markdown(f'**Experience:** {years:g} years')
        st.markdown(f'**Role:** {job_title}')
        st.markdown('</div>', unsafe_allow_html=True)

    st.success('Salary prediction generated successfully.')
else:
    c1, c2, c3 = st.columns(3)
    c1.metric('Model', 'Random Forest Regressor')
    c2.metric('R² Score', f"{metadata.get('R2 Score', 0):.3f}")
    c3.metric('Test MAE', f"{metadata.get('MAE', 0):,.0f}")

st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
with st.expander('📊 View evaluation metrics'):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('R² Score', f"{metadata.get('R2 Score', 0):.4f}")
    m2.metric('MAE', f"{metadata.get('MAE', 0):,.2f}")
    m3.metric('MSE', f"{metadata.get('MSE', 0):,.2f}")
    m4.metric('RMSE', f"{metadata.get('RMSE', 0):,.2f}")

if df is not None:
    st.markdown('<div class="section-title">Historical Dataset Preview</div>', unsafe_allow_html=True)
    with st.expander('🔎 Show first 10 records'):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="footer">
        <p style="margin-bottom: 8px;">⚠️ This system is a decision-support tool. Final compensation should also consider company policy, role scope, location, market benchmarks, and human HR review.</p>
        <p style="font-weight: 600;">© 2026 <span>KXNUX.ai</span> | All Rights Reserved</p>
    </div>
    """,
    unsafe_allow_html=True
)

