import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---- Page Config ----
st.set_page_config(
    page_title="FitIQ — Health Intelligence",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS for beautiful styling ----
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-size: 16px;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #e74c3c;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---- Load Data & Models ----
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

df, heart = load_data()
cal_model, heart_model, gender_encoder = load_models()

# ---- Sidebar Navigation ----
st.sidebar.image("https://img.icons8.com/emoji/96/flexed-biceps.png", width=80)
st.sidebar.title("💪 FitIQ")
st.sidebar.markdown("**Health & Fitness Intelligence**")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard",
    "🔥 Calorie Predictor",
    "❤️ Heart Risk Analyzer",
    "📊 Data Explorer"
])

# ==============================
# PAGE 1 — DASHBOARD
# ==============================
if page == "🏠 Dashboard":
    st.title("💪 FitIQ — Health & Fitness Intelligence")
    st.markdown("*Your personal AI-powered health analytics platform*")
    st.divider()

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Users Analyzed", f"{df['User_ID'].nunique():,}")
    col2.metric("🏋️ Workouts Tracked", f"{len(df):,}")
    col3.metric("🔥 Avg Calories Burned", f"{df['Calories'].mean():.0f}")
    col4.metric("💓 Avg Heart Rate", f"{df['Heart_Rate'].mean():.0f} bpm")

    st.divider()

    # Chart 1 - Calories by Gender
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Calories Burned by Gender")
        fig1 = px.box(df, x='Gender', y='Calories',
                      color='Gender',
                      color_discrete_map={'male': '#3498db', 'female': '#e91e8c'},
                      title='Calorie Distribution by Gender')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("⏱️ Duration vs Calories")
        fig2 = px.scatter(df, x='Duration', y='Calories',
                          color='Gender', opacity=0.6,
                          color_discrete_map={'male': '#3498db', 'female': '#e91e8c'},
                          title='Workout Duration vs Calories Burned')
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 2 - Age Distribution
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("👤 Age Distribution of Users")
        fig3 = px.histogram(df, x='Age', nbins=20,
                            color='Gender',
                            color_discrete_map={'male': '#3498db', 'female': '#e91e8c'},
                            title='User Age Distribution')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("💓 Heart Rate vs Calories")
        fig4 = px.scatter(df, x='Heart_Rate', y='Calories',
                          color='Gender', opacity=0.6,
                          color_discrete_map={'male': '#3498db', 'female': '#e91e8c'},
                          title='Heart Rate vs Calories Burned')
        st.plotly_chart(fig4, use_container_width=True)

    # Heart disease summary
    st.divider()
    st.subheader("❤️ Heart Disease Dataset Overview")
    col5, col6 = st.columns(2)

    with col5:
        hd_counts = heart['target'].value_counts().reset_index()
        hd_counts.columns = ['Status', 'Count']
        hd_counts['Status'] = hd_counts['Status'].map(
            {0: 'No Disease', 1: 'Has Disease'})
        fig5 = px.pie(hd_counts, values='Count', names='Status',
                      color_discrete_sequence=['#2ecc71', '#e74c3c'],
                      title='Heart Disease Distribution')
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.box(heart, x='target', y='age',
                      color='target',
                      color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                      labels={'target': 'Heart Disease', 'age': 'Age'},
                      title='Age Distribution by Heart Disease Status')
        st.plotly_chart(fig6, use_container_width=True)

# ==============================
# PAGE 2 — CALORIE PREDICTOR
# ==============================
elif page == "🔥 Calorie Predictor":
    st.title("🔥 Calorie Burn Predictor")
    st.markdown("*Enter your workout details to predict calories burned*")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Details")
        gender = st.selectbox("Gender", ["male", "female"])
        age = st.slider("Age", 15, 80, 25)
        height = st.slider("Height (cm)", 140, 210, 170)
        weight = st.slider("Weight (kg)", 40, 150, 70)

    with col2:
        st.subheader("🏋️ Workout Details")
        duration = st.slider("Workout Duration (minutes)", 5, 120, 30)
        heart_rate = st.slider("Average Heart Rate (bpm)", 60, 200, 120)
        body_temp = st.slider("Body Temperature (°C)", 36.0, 42.0, 38.0, 0.1)

    st.divider()

    if st.button("🔥 Predict Calories Burned"):
        # Encode gender
        gender_encoded = gender_encoder.transform([gender])[0]

        # Make prediction
        input_data = np.array([[gender_encoded, age, height, weight,
                                duration, heart_rate, body_temp]])
        prediction = cal_model.predict(input_data)[0]

        # Display result
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: #e74c3c; text-align: center;">
                🔥 Estimated Calories Burned
            </h2>
            <h1 style="color: white; text-align: center; font-size: 60px;">
                {prediction:.0f} kcal
            </h1>
            <p style="color: #aaa; text-align: center;">
                Based on your {duration} minute workout
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Health tip based on calories
        st.divider()
        if prediction < 200:
            st.info("💡 Light workout! Consider increasing duration or intensity.")
        elif prediction < 400:
            st.success("✅ Good workout! You're burning a solid amount of calories.")
        else:
            st.success("🏆 Excellent workout! You're burning a high amount of calories!")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Calories Burned"},
            gauge={
                'axis': {'range': [0, 800]},
                'bar': {'color': "#e74c3c"},
                'steps': [
                    {'range': [0, 200], 'color': "#2ecc71"},
                    {'range': [200, 400], 'color': "#f39c12"},
                    {'range': [400, 800], 'color': "#e74c3c"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================
# PAGE 3 — HEART RISK ANALYZER
# ==============================
elif page == "❤️ Heart Risk Analyzer":
    st.title("❤️ Heart Disease Risk Analyzer")
    st.markdown("*Enter your health vitals to assess heart disease risk*")
    st.warning("⚠️ This is for educational purposes only. Always consult a doctor.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Basic Info")
        age = st.slider("Age", 20, 80, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type",
                          [0, 1, 2, 3],
                          help="0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")

    with col2:
        st.subheader("🩺 Vitals")
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                           [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

    with col3:
        st.subheader("💓 Heart Metrics")
        thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina",
                             [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak ST Segment", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    st.divider()

    sex_encoded = 1 if sex == "Male" else 0

    if st.button("❤️ Analyze Heart Risk"):
        input_data = np.array([[age, sex_encoded, cp, trestbps, chol,
                                fbs, restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])
        prediction = heart_model.predict(input_data)[0]
        probability = heart_model.predict_proba(input_data)[0]
        risk_pct = probability[1] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color:#e74c3c; text-align:center;">
                    ⚠️ Elevated Heart Disease Risk Detected
                </h2>
                <h1 style="color:white; text-align:center; font-size:50px;">
                    {risk_pct:.1f}% Risk
                </h1>
                <p style="color:#aaa; text-align:center;">
                    Please consult a cardiologist for proper evaluation
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color:#2ecc71; text-align:center;">
                    ✅ Low Heart Disease Risk
                </h2>
                <h1 style="color:white; text-align:center; font-size:50px;">
                    {risk_pct:.1f}% Risk
                </h1>
                <p style="color:#aaa; text-align:center;">
                    Keep maintaining your healthy lifestyle!
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            title={'text': "Heart Disease Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c" if prediction == 1 else "#2ecc71"},
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"},
                    {'range': [30, 60], 'color': "#f39c12"},
                    {'range': [60, 100], 'color': "#e74c3c"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================
# PAGE 4 — DATA EXPLORER
# ==============================
elif page == "📊 Data Explorer":
    st.title("📊 Raw Data Explorer")
    st.divider()

    dataset = st.radio("Select Dataset",
                       ["Exercise & Calories", "Heart Disease"],
                       horizontal=True)

    if dataset == "Exercise & Calories":
        st.subheader("🏋️ Exercise & Calories Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Male Users", f"{(df['Gender']=='male').sum():,}")
        col3.metric("Female Users", f"{(df['Gender']=='female').sum():,}")
        st.dataframe(df, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Data", csv,
                           "exercise_calories.csv", "text/csv")
    else:
        st.subheader("❤️ Heart Disease Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(heart):,}")
        col2.metric("Has Disease", f"{(heart['target']==1).sum()}")
        col3.metric("No Disease", f"{(heart['target']==0).sum()}")
        st.dataframe(heart, use_container_width=True)

        csv = heart.to_csv(index=False)
        st.download_button("📥 Download Data", csv,
                           "heart_disease.csv", "text/csv")

# ---- Footer ----
st.sidebar.divider()
st.sidebar.markdown("Built with ❤️ using Python & Streamlit")
st.sidebar.markdown("📊 Data Source: Kaggle")