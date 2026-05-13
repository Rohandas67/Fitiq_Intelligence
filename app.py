import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="FitIQ Intelligence",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] * {
    color: white;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0;
    color: white;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.3rem;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    transition: 0.3s ease;
    margin-bottom: 10px;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 5px;
}

.metric-value {
    color: white;
    font-size: 32px;
    font-weight: 700;
}

.chart-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 22px;
    padding: 20px;
    margin-bottom: 20px;
}

.prediction-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.12));
    border: 1px solid rgba(255,255,255,0.08);
    padding: 35px;
    border-radius: 24px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
    margin-top: 20px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    font-weight: 600;
    font-size: 16px;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
}

.stSelectbox, .stSlider {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 6px;
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    exercise = pd.read_csv('data/exercise.csv')
    calories = pd.read_csv('data/calories.csv')
    df = pd.merge(exercise, calories, on='User_ID')

    heart = pd.read_csv('data/heart.csv')

    return df, heart


@st.cache_resource
def load_models():
    cal_model = joblib.load('models/calorie_model.pkl')
    heart_model = joblib.load('models/heart_model.pkl')
    gender_encoder = joblib.load('models/gender_encoder.pkl')

    return cal_model, heart_model, gender_encoder


plot_template = "plotly_dark"

df, heart = load_data()
cal_model, heart_model, gender_encoder = load_models()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("""
<div style="text-align:center; padding: 10px 0 25px 0;">
    <h1 style="margin-bottom:0; color:white;">FitIQ</h1>
    <p style="color:#94a3b8; margin-top:0;">
        Health Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔥 Calorie Predictor",
        "❤️ Heart Risk Analyzer",
        "📊 Data Explorer"
    ]
)

# ==========================================
# DASHBOARD
# ==========================================
if page == "🏠 Dashboard":

    st.markdown("""
    <div style="padding-bottom:25px;">
        <div class="hero-title">FitIQ Intelligence</div>
        <div class="hero-subtitle">
            AI-powered health analytics platform designed for smarter fitness and wellness insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # METRICS
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("👥 Users Analyzed", f"{df['User_ID'].nunique():,}"),
        ("🏋️ Workouts Tracked", f"{len(df):,}"),
        ("🔥 Avg Calories", f"{df['Calories'].mean():.0f}"),
        ("💓 Avg Heart Rate", f"{df['Heart_Rate'].mean():.0f} bpm")
    ]

    for col, metric in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{metric[0]}</div>
                <div class="metric-value">{metric[1]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CHARTS
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.box(
            df,
            x='Gender',
            y='Calories',
            color='Gender',
            template=plot_template,
            title='Calorie Burn Distribution by Gender'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df,
            x='Duration',
            y='Calories',
            color='Gender',
            opacity=0.7,
            template=plot_template,
            title='Workout Duration vs Calories Burned'
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(
            df,
            x='Age',
            color='Gender',
            nbins=25,
            template=plot_template,
            title='User Age Distribution'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.scatter(
            df,
            x='Heart_Rate',
            y='Calories',
            color='Gender',
            opacity=0.7,
            template=plot_template,
            title='Heart Rate vs Calories Burned'
        )
        st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# CALORIE PREDICTOR
# ==========================================
elif page == "🔥 Calorie Predictor":

    st.markdown("""
    <div style="padding-bottom:25px;">
        <div class="hero-title">Calorie Burn Predictor</div>
        <div class="hero-subtitle">
            Estimate calorie expenditure using AI-powered workout analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Details")

        gender = st.selectbox("Gender", ["male", "female"])
        age = st.slider("Age", 15, 80, 24)
        height = st.slider("Height (cm)", 140, 210, 172)
        weight = st.slider("Weight (kg)", 40, 150, 70)

    with col2:
        st.subheader("🏋️ Workout Details")

        duration = st.slider("Workout Duration", 5, 120, 30)
        heart_rate = st.slider("Average Heart Rate", 60, 200, 120)
        body_temp = st.slider("Body Temperature", 36.0, 42.0, 38.0, 0.1)

    if st.button("🔥 Generate Prediction"):

        gender_encoded = gender_encoder.transform([gender])[0]

        input_data = np.array([[gender_encoded, age, height, weight,
                                duration, heart_rate, body_temp]])

        prediction = cal_model.predict(input_data)[0]

        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="text-align:center; color:#c4b5fd;">
                Estimated Calories Burned
            </h2>

            <h1 style="text-align:center; font-size:72px; color:white; margin-bottom:0;">
                {prediction:.0f}
            </h1>

            <p style="text-align:center; color:#94a3b8; font-size:18px;">
                kcal during your {duration}-minute session
            </p>
        </div>
        """, unsafe_allow_html=True)

        if prediction < 200:
            st.info("Your current session reflects a lighter workout intensity.")

        elif prediction < 400:
            st.success("You're maintaining a balanced and effective calorie burn.")

        else:
            st.success("Strong workout intensity detected with high energy expenditure.")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Calories Burned"},
            gauge={
                'axis': {'range': [0, 800]},
                'bar': {'color': '#8b5cf6'},
                'steps': [
                    {'range': [0, 200], 'color': '#16a34a'},
                    {'range': [200, 400], 'color': '#f59e0b'},
                    {'range': [400, 800], 'color': '#ef4444'}
                ]
            }
        ))

        fig.update_layout(template=plot_template)

        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# HEART RISK ANALYZER
# ==========================================
elif page == "❤️ Heart Risk Analyzer":

    st.markdown("""
    <div style="padding-bottom:25px;">
        <div class="hero-title">Heart Risk Analyzer</div>
        <div class="hero-subtitle">
            Evaluate cardiovascular indicators using predictive health intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.warning("Educational use only. Please consult a certified medical professional for diagnosis.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 80, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

    with col2:
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])

    with col3:
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
        if st.button("❤️ Analyze Risk"):

        sex_encoded = 1 if sex == "Male" else 0

        input_data = np.array([[age, sex_encoded, cp, trestbps, chol,
                                fbs, restecg, thalach, exang,
                                oldpeak, slope, ca, thal]])

        prediction = heart_model.predict(input_data)[0]
        probability = heart_model.predict_proba(input_data)[0]

        risk_pct = probability[1] * 100

        if prediction == 1:

            title = "Elevated Cardiovascular Risk Detected"
            subtitle = "Professional medical consultation is strongly recommended."
            color = "#ef4444"

        else:

            title = "Relatively Stable Cardiovascular Indicators"
            subtitle = "Current health indicators appear comparatively balanced."
            color = "#16a34a"

        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="text-align:center; color:{color};">
                {title}
            </h2>

            <h1 style="text-align:center; color:white; font-size:70px;">
                {risk_pct:.1f}%
            </h1>

            <p style="text-align:center; color:#94a3b8;">
                {subtitle}
            </p>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={'text': "Heart Risk Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': '#16a34a'},
                    {'range': [30, 60], 'color': '#f59e0b'},
                    {'range': [60, 100], 'color': '#ef4444'}
                ]
            }
        ))

        fig.update_layout(template=plot_template)

        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# DATA EXPLORER
# ==========================================
elif page == "📊 Data Explorer":

    st.markdown("""
    <div style="padding-bottom:25px;">
        <div class="hero-title">Data Explorer</div>
        <div class="hero-subtitle">
            Explore the datasets powering FitIQ Intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    dataset = st.radio(
        "Select Dataset",
        ["Exercise & Calories", "Heart Disease"],
        horizontal=True
    )

    if dataset == "Exercise & Calories":

        st.subheader("Exercise & Calories Dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Male Users", f"{(df['Gender']=='male').sum():,}")
        col3.metric("Female Users", f"{(df['Gender']=='female').sum():,}")

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)

        st.download_button(
            "📥 Download Dataset",
            csv,
            "exercise_calories.csv",
            "text/csv"
        )

    else:

        st.subheader("Heart Disease Dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", f"{len(heart):,}")
        col2.metric("High Risk Cases", f"{(heart['target']==1).sum():,}")
        col3.metric("Low Risk Cases", f"{(heart['target']==0).sum():,}")

        st.dataframe(heart, use_container_width=True)

        csv = heart.to_csv(index=False)

        st.download_button(
            "📥 Download Dataset",
            csv,
            "heart_disease.csv",
            "text/csv"
        )

# ==========================================
# FOOTER
# ==========================================
st.sidebar.divider()

st.sidebar.markdown("""
<div style="font-size:14px; color:#94a3b8; text-align:center;">
Built with Streamlit · Machine Learning · Plotly
<br><br>
FitIQ Intelligence © 2026
</div>
""", unsafe_allow_html=True)