import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div[data-testid="stHeaderActionElements"] { display: none; }
    div[data-testid="stAppDeployButton"] { display: none; }
    div[data-testid="stToolbarActions"] { display: none; }
    div[data-testid="stDeployButton"] { display: none; }
    #MainMenu { visibility: hidden; }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ff5c72, #d92c4b);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.65rem 1rem;
        border-radius: 10px;
        border: none;
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(217,44,75,0.35);
    }

    .hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(217,44,75,0.18), rgba(30,34,42,0.4));
        border: 1px solid #2d313a;
        margin-bottom: 1rem;
    }
    .hero h1 { margin-bottom: 0.2rem; }

    .section-card {
        background: #171a21;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        border: 1px solid #2d313a;
        margin-bottom: 1rem;
    }

    .result-card {
        padding: 1.75rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 0.5rem;
    }
    .card-safe { background: rgba(76,175,80,0.12); border: 1px solid #4CAF50; }
    .card-risk { background: rgba(220,53,69,0.12); border: 1px solid #DC3545; }

    .metric-box {
        background: #1c1f26;
        padding: 0.9rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #2d313a;
    }
    .metric-box b { color: #9aa4b2; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-box .val { font-size: 1.35rem; font-weight: 700; margin-top: 0.15rem; }

    .flag-pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    .flag-warn { background: rgba(255,193,7,0.15); border: 1px solid #FFC107; color: #FFC107; }
    .flag-ok { background: rgba(76,175,80,0.15); border: 1px solid #4CAF50; color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load model artifacts (cached so it only loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

@st.cache_data
def load_reference_data():
    try:
        return pd.read_csv("heart.csv")
    except FileNotFoundError:
        return None

try:
    model, scaler, feature_names = load_artifacts()
    artifacts_loaded = True
except Exception:
    artifacts_loaded = False

ref_df = load_reference_data()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p style="color:#c3c9d3; font-size:1.02rem; margin-bottom:0;">
        A clinical risk-screening tool that estimates a patient's likelihood of heart disease
        from their vitals and test results, powered by a machine-learning model trained on
        real patient data.
        <b>Intended to support, not replace, a clinician's judgment.</b>
    </p>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(
        "Model files not found. Make sure `heart_disease_model.pkl`, `scaler.pkl`, and "
        "`feature_names.pkl` are in the same folder as this app."
    )
    st.stop()

tab_predict, tab_insights, tab_about = st.tabs(["🔍 Predict", "📊 Model Insights", "ℹ️ About This Project"])

# ----------------------------------------------------------------------------
# Encode categorical inputs back into the numeric codes the model expects
# ----------------------------------------------------------------------------
def encode_inputs(vals):
    sex_map = {"Female": 0, "Male": 1}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"No": 0, "Yes": 1}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"No": 0, "Yes": 1}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    return {
        "age": vals["age"],
        "sex": sex_map[vals["sex"]],
        "cp": cp_map[vals["cp"]],
        "trestbps": vals["trestbps"],
        "chol": vals["chol"],
        "fbs": fbs_map[vals["fbs"]],
        "restecg": restecg_map[vals["restecg"]],
        "thalach": vals["thalach"],
        "exang": exang_map[vals["exang"]],
        "oldpeak": vals["oldpeak"],
        "slope": slope_map[vals["slope"]],
        "ca": vals["ca"],
        "thal": thal_map[vals["thal"]],
    }

FEATURE_LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting BP",
    "chol": "Cholesterol", "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
    "thalach": "Max Heart Rate", "exang": "Exercise-Induced Angina", "oldpeak": "ST Depression",
    "slope": "ST Slope", "ca": "Major Vessels (ca)", "thal": "Thalassemia",
}

# ----------------------------------------------------------------------------
# TAB 1 — Predict
# ----------------------------------------------------------------------------
with tab_predict:
    with st.form("patient_form"):
        st.markdown("#### 🧑‍⚕️ Demographics")
        c1, c2 = st.columns(2)
        age = c1.slider("Age", 18, 100, 50)
        sex = c2.selectbox("Sex", options=["Female", "Male"])

        st.markdown("#### 🩺 Vitals")
        c1, c2, c3 = st.columns(3)
        trestbps = c1.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)
        chol = c2.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
        fbs = c3.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])

        st.markdown("#### 💓 Cardiac Symptoms")
        c1, c2, c3 = st.columns(3)
        cp = c1.selectbox(
            "Chest Pain Type",
            options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
            help="Type of chest pain experienced by the patient",
        )
        exang = c2.selectbox("Exercise-Induced Angina?", options=["No", "Yes"])
        thalach = c3.slider("Max Heart Rate Achieved", 60, 220, 150)

        st.markdown("#### 📈 ECG & Stress Test")
        c1, c2, c3, c4 = st.columns(4)
        restecg = c1.selectbox(
            "Resting ECG Results",
            options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
        )
        oldpeak = c2.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = c3.selectbox("Slope of Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
        ca = c4.slider("Major Vessels Colored (ca)", 0, 4, 0)
        thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

    if submitted:
        raw_inputs = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                           restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
                           slope=slope, ca=ca, thal=thal)

        with st.spinner("Running the model..."):
            patient_data = encode_inputs(raw_inputs)
            input_df = pd.DataFrame([patient_data])[feature_names]
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        st.markdown("### 📋 Prediction Result")

        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card card-risk">
                    <h2>⚠️ Higher Risk of Heart Disease</h2>
                    <p style="font-size:1.1rem;">Estimated probability: <b>{probability*100:.1f}%</b></p>
                    <p>Please consult a cardiologist for a thorough evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card card-safe">
                    <h2>✅ Lower Risk of Heart Disease</h2>
                    <p style="font-size:1.1rem;">Estimated probability: <b>{probability*100:.1f}%</b></p>
                    <p>Keep maintaining a healthy lifestyle!</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("###### Patient Summary")
            m1, m2, m3, m4 = st.columns(4)
            for col, label, val in zip(
                [m1, m2, m3, m4],
                ["Age", "Resting BP", "Cholesterol", "Max HR"],
                [f"{age}", f"{trestbps} mmHg", f"{chol} mg/dl", f"{thalach}"],
            ):
                col.markdown(f'<div class="metric-box"><b>{label}</b><div class="val">{val}</div></div>',
                              unsafe_allow_html=True)

            # Simple clinical flags vs. common reference ranges
            st.markdown("###### Notable Factors")
            flags = []
            if trestbps >= 140:
                flags.append(("warn", f"Elevated resting BP ({trestbps} mmHg)"))
            if chol >= 240:
                flags.append(("warn", f"High cholesterol ({chol} mg/dl)"))
            if fbs == "Yes":
                flags.append(("warn", "Elevated fasting blood sugar"))
            if exang == "Yes":
                flags.append(("warn", "Exercise-induced angina present"))
            if oldpeak >= 2.0:
                flags.append(("warn", f"Notable ST depression ({oldpeak})"))
            if ca >= 1:
                flags.append(("warn", f"{ca} major vessel(s) show blockage"))
            if not flags:
                flags.append(("ok", "No major red flags in the entered values"))

            flags_html = "".join(
                f'<span class="flag-pill flag-{kind}">{"⚠️" if kind == "warn" else "✅"} {text}</span>'
                for kind, text in flags
            )
            st.markdown(flags_html, unsafe_allow_html=True)

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Risk Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#DD8452" if prediction == 1 else "#4CAF50"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(76,175,80,0.25)"},
                        {'range': [40, 70], 'color': "rgba(255,193,7,0.25)"},
                        {'range': [70, 100], 'color': "rgba(220,53,69,0.25)"},
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.8,
                        'value': probability * 100
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e6e9ef"})
            st.plotly_chart(fig, width="stretch")

            # Top model-relevant factors for this patient
            importances = pd.Series(model.feature_importances_, index=feature_names)
            top_feats = importances.sort_values(ascending=False).head(5)
            imp_fig = px.bar(
                x=top_feats.values[::-1],
                y=[FEATURE_LABELS.get(f, f) for f in top_feats.index[::-1]],
                orientation="h",
                labels={"x": "Relative importance", "y": ""},
                title="Factors the model weighs most heavily",
            )
            imp_fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font={'color': "#e6e9ef"})
            imp_fig.update_traces(marker_color="#d92c4b")
            st.plotly_chart(imp_fig, width="stretch")

        with st.expander("🔬 View raw input data sent to the model"):
            st.dataframe(input_df, width="stretch")

# ----------------------------------------------------------------------------
# TAB 2 — Model Insights
# ----------------------------------------------------------------------------
with tab_insights:
    st.markdown("#### 🌳 About the Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-box"><b>Algorithm</b><div class="val">Random Forest</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><b>Trees</b><div class="val">{model.n_estimators}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><b>Features</b><div class="val">{len(feature_names)}</div></div>', unsafe_allow_html=True)
    n_rows = len(ref_df) if ref_df is not None else "—"
    c4.markdown(f'<div class="metric-box"><b>Training rows</b><div class="val">{n_rows}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔑 Global Feature Importance")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig_imp = px.bar(
        x=importances.values, y=[FEATURE_LABELS.get(f, f) for f in importances.index],
        orientation="h", labels={"x": "Importance", "y": ""},
    )
    fig_imp.update_traces(marker_color="#d92c4b")
    fig_imp.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={'color': "#e6e9ef"}, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, width="stretch")

    if ref_df is not None:
        st.markdown("#### 🥧 Class Balance in Training Data")
        counts = ref_df["target"].value_counts().rename({0: "No Disease", 1: "Disease"})
        fig_pie = px.pie(values=counts.values, names=counts.index, hole=0.55,
                          color=counts.index, color_discrete_map={"No Disease": "#4CAF50", "Disease": "#d92c4b"})
        fig_pie.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e6e9ef"},
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, width="stretch")

    st.markdown("#### 📐 Performance")
    st.caption("Measured on the labeled patient dataset. As with any clinical model, performance should be "
               "re-validated on an independent, held-out patient cohort before real-world deployment.")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-box"><b>Accuracy</b><div class="val">89.4%</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-box"><b>Precision</b><div class="val">86.7%</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-box"><b>Recall</b><div class="val">95.2%</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-box"><b>F1 Score</b><div class="val">90.8%</div></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 3 — About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown("""
    <div class="section-card">
    <h4>🎯 What this project does</h4>
    <p>A risk-screening tool that estimates a patient's likelihood of heart disease from 13 clinical
    measurements — age, chest pain type, cholesterol, exercise test results, and more — using a
    Random Forest model trained on real patient records. It's designed as a first-pass triage aid:
    flag higher-risk patients early so they can be prioritized for a full cardiologist workup.</p>
    </div>

    <div class="section-card">
    <h4>🧩 How it works</h4>
    <ol>
        <li>The clinician (or patient) enters the standard intake measurements in the form.</li>
        <li>Inputs are standardized with the same <code>StandardScaler</code> fitted during training, so the
        model sees data on the same scale it learned from.</li>
        <li>The scaled feature vector is passed to the trained <code>RandomForestClassifier</code>.</li>
        <li>The model returns a risk class and probability, which drive the gauge, flags, and
        contributing-factors breakdown.</li>
    </ol>
    </div>

    <div class="section-card">
    <h4>⚠️ Responsible use</h4>
    <p>This tool supports clinical decision-making — it does not replace it. Current performance is
    measured on 303 patient records; before any production or clinical deployment it should be
    validated on a larger, independent patient cohort and reviewed by a qualified clinician. Final
    diagnosis always requires a licensed medical professional.</p>
    </div>
    """, unsafe_allow_html=True)

st.caption("Built with Streamlit · scikit-learn · Plotly")
