import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------> PAGE CONFIG
st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------> CUSTOM CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }

    /* Title */
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #ff4b4b, #ff8e53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.3rem !important;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #cfcfcf;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }

    /* Card style for inputs */
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.85rem;
        font-size: 1.15rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
        background: linear-gradient(90deg, #ff3333, #ff5555);
    }

    /* Result boxes */
    .success-box {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 176, 155, 0.3);
        margin-top: 1.5rem;
    }
    .error-box {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 8px 25px rgba(255, 65, 108, 0.35);
        margin-top: 1.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.07);
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------> LOAD MODEL 

@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# ----------------------------------------------------> HEADER 
st.markdown("<h1>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your health details below to assess your risk of heart disease</p>', unsafe_allow_html=True)

# ------------------------------------------------------->INPUT FORM 
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Personal Info")
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")

    with col2:
        st.markdown("### 🩺 Clinical Measurements")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)

    with col3:
        st.markdown("### ❤️ Cardiac Indicators")
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No")

# Second row of inputs
col4, col5, col6 = st.columns(3)

with col4:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {
            "ATA": "Atypical Angina",
            "NAP": "Non-Anginal Pain",
            "TA": "Typical Angina",
            "ASY": "Asymptomatic"
        }[x]
    )

with col5:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST-T Wave Abnormality",
            "LVH": "Left Ventricular Hypertrophy"
        }[x]
    )

with col6:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"],
    format_func=lambda x: {
        "Up": "Upsloping",
        "Flat": "Flat",
        "Down": "Downsloping"
    }[x]
)

st.write("")  # spacing

# -----------------------------------------------------------> PREDICT BUTTON 
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_clicked = st.button("🔍 Predict Heart Disease Risk")

# -------------------------------------------------------------> PREDICTION LOGIC 
if predict_clicked:
    # Create raw input
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Optional: get probability if available
    try:
        proba = model.predict_proba(scaled_input)[0]
        risk_score = proba[1] * 100
    except:
        risk_score = None

    # ------------------------------------------------> RESULT DISPLAY 
    st.markdown("---")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="error-box">
            ⚠️ High Risk of Heart Disease
            {f'<br><span style="font-size:1.1rem; opacity:0.9;">Estimated Risk: {risk_score:.1f}%</span>' if risk_score else ''}
        </div>
        """, unsafe_allow_html=True)
        st.warning("Please consult a cardiologist for further evaluation. This is not a medical diagnosis.")
    else:
        st.markdown(f"""
        <div class="success-box">
            ✅ Low Risk of Heart Disease
            {f'<br><span style="font-size:1.1rem; opacity:0.9;">Estimated Risk: {risk_score:.1f}%</span>' if risk_score else ''}
        </div>
        """, unsafe_allow_html=True)
        st.info("Keep maintaining a healthy lifestyle!")

# --------------------------------------------------------------------> FOOTER 
st.markdown("---")
st.markdown("""
<div class="footer-note">
    Built with ❤️ using Streamlit • For educational purposes only • Not a substitute for professional medical advice<br>
    <em>AI-assisted UI • Learning project from a tutorial</em>
</div>
""", unsafe_allow_html=True)