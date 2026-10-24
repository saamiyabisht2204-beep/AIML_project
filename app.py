"""
Interactive Streamlit Web Application for AI-Based Student Stress Level Prediction.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .verdict-high {
        padding: 1.2rem;
        background-color: #FEE2E2;
        border-left: 6px solid #DC2626;
        border-radius: 8px;
        color: #991B1B;
    }
    .verdict-low {
        padding: 1.2rem;
        background-color: #DCFCE7;
        border-left: 6px solid #16A34A;
        border-radius: 8px;
        color: #166534;
    }
    .verdict-moderate {
        padding: 1.2rem;
        background-color: #FEF3C7;
        border-left: 6px solid #D97706;
        border-radius: 8px;
        color: #92400E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎓 Student Stress Level Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter academic and lifestyle factors to predict whether a student is experiencing <b>High Stress</b>, <b>Moderate Stress</b>, or <b>Low Stress</b>.</div>',
    unsafe_allow_html=True,
)

# Ensure model artifacts exist or train automatically
model_file = Path("models/best_model.joblib")
prep_file = Path("models/preprocessor.joblib")

@st.cache_resource(show_spinner="Training machine learning models...")
def load_or_train_predictor():
    if not (model_file.exists() and prep_file.exists()):
        from src.models.train import train_and_evaluate_all
        train_and_evaluate_all()
    from src.models.predict import StressPredictor
    return StressPredictor()

try:
    predictor = load_or_train_predictor()
except Exception as e:
    st.error(f"Error initializing ML model: {e}")
    st.stop()

# Sidebar: Quick presets
st.sidebar.header("⚡ Quick Preset Profiles")
preset = st.sidebar.radio(
    "Load sample student profile:",
    ("Custom Input", "Typical High Stress Student", "Typical Low Stress Student", "Balanced Student"),
)

# Default values based on presets
if preset == "Typical High Stress Student":
    default_study = 9.0
    default_extra = 1.0
    default_sleep = 5.2
    default_social = 1.5
    default_physical = 0.5
    default_gpa = 3.6
elif preset == "Typical Low Stress Student":
    default_study = 4.5
    default_extra = 3.0
    default_sleep = 8.5
    default_social = 4.0
    default_physical = 3.0
    default_gpa = 2.8
elif preset == "Balanced Student":
    default_study = 6.0
    default_extra = 2.0
    default_sleep = 7.5
    default_social = 2.5
    default_physical = 2.0
    default_gpa = 3.1
else:
    default_study = 6.0
    default_extra = 2.0
    default_sleep = 7.0
    default_social = 2.5
    default_physical = 1.5
    default_gpa = 3.0

tabs = st.tabs(["📝 Predict Student Stress", "📁 Batch CSV Scoring", "📈 Model Benchmarks"])

# ----------------- TAB 1: SINGLE PREDICTION -----------------
with tabs[0]:
    st.subheader("Student Factors Input Form")
    st.write("Adjust the sliders or type values directly into the input fields:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📚 Academic Factors")
            study_hours = st.number_input(
                "Study Hours Per Day (0 - 16 hrs)",
                min_value=0.0,
                max_value=16.0,
                value=float(default_study),
                step=0.2,
                help="Total hours spent studying outside of classes.",
            )
            gpa = st.number_input(
                "Cumulative GPA (0.0 - 4.0)",
                min_value=0.0,
                max_value=4.0,
                value=float(default_gpa),
                step=0.05,
                help="Academic performance metric.",
            )
            extracurricular_hours = st.number_input(
                "Extracurricular Hours Per Day (0 - 10 hrs)",
                min_value=0.0,
                max_value=10.0,
                value=float(default_extra),
                step=0.2,
                help="Clubs, projects, and non-academic commitments.",
            )

        with col2:
            st.markdown("#### 🏃 Lifestyle & Well-being")
            sleep_hours = st.number_input(
                "Sleep Duration (0 - 14 hrs/day)",
                min_value=0.0,
                max_value=14.0,
                value=float(default_sleep),
                step=0.2,
                help="Average daily sleep hours.",
            )
            physical_activity = st.number_input(
                "Physical Activity / Exercise (0 - 8 hrs/day)",
                min_value=0.0,
                max_value=8.0,
                value=float(default_physical),
                step=0.2,
                help="Hours spent exercising or in sports.",
            )
            social_hours = st.number_input(
                "Social Activity Hours (0 - 12 hrs/day)",
                min_value=0.0,
                max_value=12.0,
                value=float(default_social),
                step=0.2,
                help="Time spent with friends, family, and peers.",
            )

        submit_btn = st.form_submit_button("🔍 Run Stress Prediction", type="primary", use_container_width=True)

    if submit_btn:
        input_data = {
            "Study_Hours_Per_Day": study_hours,
            "Extracurricular_Hours_Per_Day": extracurricular_hours,
            "Sleep_Hours_Per_Day": sleep_hours,
            "Social_Hours_Per_Day": social_hours,
            "Physical_Activity_Hours_Per_Day": physical_activity,
            "GPA": gpa,
        }

        result = predictor.predict_single(input_data)
        stress_level = result["predicted_stress_level"]
        confidence = result["confidence"]
        probabilities = result["class_probabilities"]

        st.markdown("---")
        st.subheader("🎯 Prediction Verdict")

        # Distinct Verdict Display
        if stress_level == "High":
            st.markdown(
                f"""
                <div class="verdict-high">
                    <h2>🚨 VERDICT: HIGH STRESS</h2>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        The machine learning model determined that this student is at <b>HIGH RISK of stress</b> (Confidence: <b>{confidence * 100:.1f}%</b>).
                    </p>
                    <b>Recommended Actions:</b>
                    <ul>
                        <li>Reduce consecutive study blocks and incorporate mandatory rest intervals.</li>
                        <li>Target a minimum of 7-8 hours of sleep per night.</li>
                        <li>Consider reaching out to an academic mentor or student counselor.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif stress_level == "Low":
            st.markdown(
                f"""
                <div class="verdict-low">
                    <h2>🟢 VERDICT: LOW STRESS</h2>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        The machine learning model determined that this student has a <b>LOW / HEALTHY stress profile</b> (Confidence: <b>{confidence * 100:.1f}%</b>).
                    </p>
                    <b>Assessment:</b>
                    <ul>
                        <li>Balanced distribution between study hours, sleep, and physical activity.</li>
                        <li>Maintain this routine during exam cycles.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-moderate">
                    <h2>⚠️ VERDICT: MODERATE STRESS</h2>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        The machine learning model determined that this student is experiencing <b>MODERATE stress</b> (Confidence: <b>{confidence * 100:.1f}%</b>).
                    </p>
                    <b>Assessment:</b>
                    <ul>
                        <li>Moderate workload detected. Monitor sleep consistency to prevent progression to high stress.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probabilities & Input Recap
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("#### 📊 Model Probability Distribution")
            prob_df = pd.DataFrame(
                list(probabilities.items()),
                columns=["Stress Category", "Probability"],
            )
            st.bar_chart(prob_df.set_index("Stress Category"))

        with res_col2:
            st.markdown("#### 📋 Evaluated Student Profile")
            recap_df = pd.DataFrame(
                [
                    {"Factor": "Study Hours / Day", "Value": f"{study_hours} hrs"},
                    {"Factor": "Sleep Duration", "Value": f"{sleep_hours} hrs"},
                    {"Factor": "Physical Activity", "Value": f"{physical_activity} hrs"},
                    {"Factor": "Social Hours", "Value": f"{social_hours} hrs"},
                    {"Factor": "Extracurriculars", "Value": f"{extracurricular_hours} hrs"},
                    {"Factor": "GPA", "Value": f"{gpa}"},
                ]
            )
            st.dataframe(recap_df, hide_index=True, use_container_width=True)

# ----------------- TAB 2: BATCH CSV -----------------
with tabs[1]:
    st.subheader("Bulk Student Data Scoring")
    st.write("Upload a CSV file containing multiple student entries to predict stress levels for the entire batch.")

    sample_template = pd.DataFrame(
        [
            {
                "Student_ID": 101,
                "Study_Hours_Per_Day": 7.5,
                "Extracurricular_Hours_Per_Day": 1.5,
                "Sleep_Hours_Per_Day": 6.0,
                "Social_Hours_Per_Day": 2.0,
                "Physical_Activity_Hours_Per_Day": 1.0,
                "GPA": 3.4,
            },
            {
                "Student_ID": 102,
                "Study_Hours_Per_Day": 4.0,
                "Extracurricular_Hours_Per_Day": 2.5,
                "Sleep_Hours_Per_Day": 8.0,
                "Social_Hours_Per_Day": 3.5,
                "Physical_Activity_Hours_Per_Day": 2.5,
                "GPA": 2.9,
            },
        ]
    )
    st.download_button(
        label="📄 Download Sample CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="sample_student_input.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", batch_df.head())

        try:
            results_df = predictor.predict_batch(batch_df)
            st.success(f"Successfully processed {len(results_df)} student records!")
            st.dataframe(results_df, use_container_width=True)

            csv_download = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored CSV with Predictions",
                data=csv_download,
                file_name="student_stress_predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as err:
            st.error(f"Error processing file: {err}")

# ----------------- TAB 3: BENCHMARKS -----------------
with tabs[2]:
    st.subheader("Model Evaluation & Benchmarking")
    comp_file = Path("reports/model_comparison.csv")
    if comp_file.is_file():
        st.dataframe(pd.read_csv(comp_file), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cm_path = Path("reports/figures/decision_tree_confusion_matrix.png")
        if cm_path.is_file():
            st.image(str(cm_path), caption="Decision Tree Confusion Matrix")
    with c2:
        fi_path = Path("reports/figures/decision_tree_feature_importance.png")
        if fi_path.is_file():
            st.image(str(fi_path), caption="Feature Importances")
