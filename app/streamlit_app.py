"""
Interactive Streamlit Application for Blood Pressure Risk Prediction & Clinical Decision Support.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to sys.path to enable imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.data_loader import RISK_CATEGORIES, assign_aha_bp_category, FEATURE_COLUMNS
from src.predictor import BPRiskPredictor
from src.explainability import plot_shap_waterfall_interactive
from src.train import train_and_benchmark_models

# Streamlit Page Config
st.set_page_config(
    page_title="CardioSense | Blood Pressure Risk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .badge-normal { background-color: #DEF7EC; color: #03543F; border: 1px solid #31C48D; }
    .badge-elevated { background-color: #FEF08A; color: #854D0E; border: 1px solid #EAB308; }
    .badge-stage1 { background-color: #FFEDD5; color: #9A3412; border: 1px solid #F97316; }
    .badge-stage2 { background-color: #FEE2E2; color: #991B1B; border: 1px solid #EF4444; }
    .badge-crisis { background-color: #4C0519; color: #FFE4E6; border: 1px solid #BE123C; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    models_dir = os.path.join(PROJECT_DIR, "models")
    model_path = os.path.join(models_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        with st.spinner("Training machine learning models on clinical data..."):
            train_and_benchmark_models(n_samples=3500, output_dir=models_dir)
    return BPRiskPredictor(models_dir=models_dir)


def render_risk_badge(category: str):
    css_class_map = {
        "Normal": "badge-normal",
        "Elevated": "badge-elevated",
        "Stage 1 Hypertension": "badge-stage1",
        "Stage 2 Hypertension": "badge-stage2",
        "Hypertensive Crisis": "badge-crisis"
    }
    badge_class = css_class_map.get(category, "badge-normal")
    icon_map = {
        "Normal": "🟢",
        "Elevated": "🟡",
        "Stage 1 Hypertension": "🟠",
        "Stage 2 Hypertension": "🔴",
        "Hypertensive Crisis": "🚨"
    }
    icon = icon_map.get(category, "🩺")
    html = f'<div class="risk-badge {badge_class}">{icon} {category}</div>'
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">🩺 CardioSense: Blood Pressure Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Clinical Risk Stratification & Explainable Cardiovascular Health Awareness (AHA/ACC Aligned)</div>', unsafe_allow_html=True)

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Error initializing predictor: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "👤 Individual Patient Assessment",
        "📂 Batch CSV Triage",
        "📊 Model Benchmarks & Explainability"
    ])

    # =========================================================================
    # TAB 1: INDIVIDUAL PATIENT ASSESSMENT
    # =========================================================================
    with tab1:
        st.markdown("### 📋 Patient Clinical Profile")

        # Preset Profiles Selector
        preset = st.selectbox(
            "Quick-Fill Example Patient Profile:",
            [
                "Custom Input",
                "Healthy Athlete (Optimal Profile)",
                "Pre-hypertensive Desk Professional",
                "Stage 1 Borderline Hypertensive",
                "Stage 2 High-Risk Diabetic Patient",
                "Hypertensive Emergency Crisis"
            ]
        )

        # Default values based on preset
        defaults = {
            "age": 45, "gender": "Male", "systolic_bp": 128, "diastolic_bp": 82, "heart_rate": 74.0,
            "bmi": 26.5, "cholesterol_total": 195.0, "cholesterol_hdl": 48.0, "cholesterol_ldl": 120.0,
            "triglycerides": 145.0, "fasting_blood_sugar": 92.0, "physical_activity_hours": 3.0,
            "daily_sodium_mg": 2800.0, "smoking_status": "Non-Smoker", "alcohol_drinks_per_week": 2.0,
            "stress_level": 5, "sleep_hours": 7.0, "family_history_hypertension": 0,
            "has_diabetes": 0, "has_ckd": 0
        }

        if preset == "Healthy Athlete (Optimal Profile)":
            defaults.update({
                "age": 28, "gender": "Female", "systolic_bp": 112, "diastolic_bp": 72, "heart_rate": 58.0,
                "bmi": 21.5, "cholesterol_total": 160.0, "cholesterol_hdl": 65.0, "cholesterol_ldl": 80.0,
                "triglycerides": 85.0, "fasting_blood_sugar": 82.0, "physical_activity_hours": 7.5,
                "daily_sodium_mg": 1800.0, "smoking_status": "Non-Smoker", "alcohol_drinks_per_week": 1.0,
                "stress_level": 3, "sleep_hours": 8.0, "family_history_hypertension": 0,
                "has_diabetes": 0, "has_ckd": 0
            })
        elif preset == "Pre-hypertensive Desk Professional":
            defaults.update({
                "age": 42, "gender": "Male", "systolic_bp": 126, "diastolic_bp": 78, "heart_rate": 76.0,
                "bmi": 27.2, "cholesterol_total": 210.0, "cholesterol_hdl": 44.0, "cholesterol_ldl": 135.0,
                "triglycerides": 165.0, "fasting_blood_sugar": 98.0, "physical_activity_hours": 1.5,
                "daily_sodium_mg": 3600.0, "smoking_status": "Non-Smoker", "alcohol_drinks_per_week": 4.0,
                "stress_level": 7, "sleep_hours": 6.0, "family_history_hypertension": 1,
                "has_diabetes": 0, "has_ckd": 0
            })
        elif preset == "Stage 1 Borderline Hypertensive":
            defaults.update({
                "age": 52, "gender": "Male", "systolic_bp": 136, "diastolic_bp": 86, "heart_rate": 80.0,
                "bmi": 29.4, "cholesterol_total": 225.0, "cholesterol_hdl": 40.0, "cholesterol_ldl": 145.0,
                "triglycerides": 190.0, "fasting_blood_sugar": 108.0, "physical_activity_hours": 1.0,
                "daily_sodium_mg": 4100.0, "smoking_status": "Former Smoker", "alcohol_drinks_per_week": 6.0,
                "stress_level": 8, "sleep_hours": 5.5, "family_history_hypertension": 1,
                "has_diabetes": 0, "has_ckd": 0
            })
        elif preset == "Stage 2 High-Risk Diabetic Patient":
            defaults.update({
                "age": 63, "gender": "Female", "systolic_bp": 154, "diastolic_bp": 94, "heart_rate": 88.0,
                "bmi": 33.5, "cholesterol_total": 250.0, "cholesterol_hdl": 36.0, "cholesterol_ldl": 170.0,
                "triglycerides": 240.0, "fasting_blood_sugar": 145.0, "physical_activity_hours": 0.5,
                "daily_sodium_mg": 4800.0, "smoking_status": "Current Smoker", "alcohol_drinks_per_week": 5.0,
                "stress_level": 8, "sleep_hours": 5.0, "family_history_hypertension": 1,
                "has_diabetes": 1, "has_ckd": 0
            })
        elif preset == "Hypertensive Emergency Crisis":
            defaults.update({
                "age": 68, "gender": "Male", "systolic_bp": 188, "diastolic_bp": 122, "heart_rate": 105.0,
                "bmi": 35.0, "cholesterol_total": 280.0, "cholesterol_hdl": 30.0, "cholesterol_ldl": 195.0,
                "triglycerides": 320.0, "fasting_blood_sugar": 180.0, "physical_activity_hours": 0.0,
                "daily_sodium_mg": 5500.0, "smoking_status": "Current Smoker", "alcohol_drinks_per_week": 12.0,
                "stress_level": 10, "sleep_hours": 4.5, "family_history_hypertension": 1,
                "has_diabetes": 1, "has_ckd": 1
            })

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics & Vitals")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(defaults["age"]))
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if defaults["gender"] == "Male" else 1)
            systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=80, max_value=230, value=int(defaults["systolic_bp"]))
            diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=140, value=int(defaults["diastolic_bp"]))
            heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40.0, max_value=150.0, value=float(defaults["heart_rate"]))
            bmi = st.number_input("BMI (kg/m²)", min_value=14.0, max_value=55.0, value=float(defaults["bmi"]), step=0.1)

        with col2:
            st.markdown("#### 🧪 Biochemical Markers")
            chol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=float(defaults["cholesterol_total"]))
            chol_hdl = st.number_input("HDL Good Cholesterol (mg/dL)", min_value=15.0, max_value=110.0, value=float(defaults["cholesterol_hdl"]))
            chol_ldl = st.number_input("LDL Bad Cholesterol (mg/dL)", min_value=40.0, max_value=300.0, value=float(defaults["cholesterol_ldl"]))
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=30.0, max_value=600.0, value=float(defaults["triglycerides"]))
            fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50.0, max_value=350.0, value=float(defaults["fasting_blood_sugar"]))

        with col3:
            st.markdown("#### 🏃 Lifestyle & Medical History")
            smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Former Smoker", "Current Smoker"],
                                  index=["Non-Smoker", "Former Smoker", "Current Smoker"].index(defaults["smoking_status"]))
            activity = st.slider("Physical Activity (hrs/week)", min_value=0.0, max_value=20.0, value=float(defaults["physical_activity_hours"]), step=0.5)
            sodium = st.slider("Daily Sodium Intake (mg/day)", min_value=500, max_value=6000, value=int(defaults["daily_sodium_mg"]), step=50)
            alcohol = st.number_input("Alcohol Drinks / Week", min_value=0.0, max_value=40.0, value=float(defaults["alcohol_drinks_per_week"]), step=0.5)
            stress = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=int(defaults["stress_level"]))
            sleep = st.slider("Sleep Duration (hrs/night)", min_value=3.0, max_value=12.0, value=float(defaults["sleep_hours"]), step=0.5)

            st.markdown("**Comorbidities:**")
            family_hist = st.checkbox("Family History of Hypertension", value=bool(defaults["family_history_hypertension"]))
            has_diabetes = st.checkbox("Diagnosed with Diabetes", value=bool(defaults["has_diabetes"]))
            has_ckd = st.checkbox("Chronic Kidney Disease (CKD)", value=bool(defaults["has_ckd"]))

        patient_input = {
            "age": age,
            "gender": gender,
            "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
            "heart_rate": float(heart_rate),
            "bmi": float(bmi),
            "cholesterol_total": float(chol_total),
            "cholesterol_hdl": float(chol_hdl),
            "cholesterol_ldl": float(chol_ldl),
            "triglycerides": float(triglycerides),
            "fasting_blood_sugar": float(fbs),
            "physical_activity_hours": float(activity),
            "daily_sodium_mg": float(sodium),
            "smoking_status": smoking,
            "alcohol_drinks_per_week": float(alcohol),
            "stress_level": int(stress),
            "sleep_hours": float(sleep),
            "family_history_hypertension": int(family_hist),
            "has_diabetes": int(has_diabetes),
            "has_ckd": int(has_ckd)
        }

        # Run Prediction
        result = predictor.predict_single(patient_input)

        st.markdown("---")
        st.markdown("### 🎯 Risk Assessment Results")

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

        with res_col1:
            st.markdown("#### Predicted Clinical Category")
            render_risk_badge(result["predicted_category"])
            st.metric("Model Prediction Confidence", f"{result['confidence']*100:.1f}%")
            
            # Severity Gauge
            score = result["risk_score_100"]
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "Cardiovascular Risk Score", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1E293B"},
                    "steps": [
                        {"range": [0, 25], "color": "#10B981"},
                        {"range": [25, 50], "color": "#FBBF24"},
                        {"range": [50, 75], "color": "#F97316"},
                        {"range": [75, 100], "color": "#EF4444"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": score
                    }
                }
            ))
            gauge_fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_col2:
            st.markdown("#### 📈 Risk Probability Distribution")
            prob_df = pd.DataFrame({
                "Category": list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values())
            })
            colors = ["#10B981", "#EAB308", "#F97316", "#EF4444", "#991B1B"]
            prob_fig = px.bar(
                prob_df,
                x="Probability",
                y="Category",
                orientation="h",
                color="Category",
                color_discrete_sequence=colors,
                text_auto=".1%"
            )
            prob_fig.update_layout(
                showlegend=False,
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(range=[0, 1.05])
            )
            st.plotly_chart(prob_fig, use_container_width=True)

        with res_col3:
            st.markdown("#### 🔬 Physiological Derived Indices")
            pulse_p = systolic_bp - diastolic_bp
            map_val = diastolic_bp + (pulse_p / 3.0)
            chol_ratio = chol_total / max(chol_hdl, 1.0)

            st.metric("Pulse Pressure (SBP - DBP)", f"{pulse_p:.0f} mmHg",
                      help="Normal is ~40 mmHg. >60 mmHg indicates arterial stiffness.")
            st.metric("Mean Arterial Pressure (MAP)", f"{map_val:.1f} mmHg",
                      help="Average arterial pressure during a cardiac cycle. Normal: 70-100 mmHg.")
            st.metric("Cholesterol Ratio (Total/HDL)", f"{chol_ratio:.2f}",
                      help="Ideal ratio is below 3.5. Above 5.0 indicates elevated risk.")

        # SHAP Waterfall & Recommendations
        st.markdown("---")
        exp_col, rec_col = st.columns([1.1, 1.1])

        with exp_col:
            st.markdown("### 🔍 Explainable AI: Feature Attributions (SHAP)")
            st.caption("Shows which patient factors contributed most to raising (Red) or lowering (Green) the risk score.")
            if "top_features" in result["explanation"] and result["explanation"]["top_features"]:
                shap_fig = plot_shap_waterfall_interactive(
                    result["explanation"]["top_features"],
                    result["explanation"]["base_value"],
                    result["predicted_category"]
                )
                st.plotly_chart(shap_fig, use_container_width=True)
            else:
                st.info("SHAP explanation not available for this configuration.")

        with rec_col:
            st.markdown("### 💡 Evidence-Based Clinical & Lifestyle Guidance")
            st.caption("Personalized recommendations based on AHA/ACC cardiovascular health guidelines.")
            for rec in result["recommendations"]:
                priority_color = "#EF4444" if rec["priority"] == "CRITICAL" else (
                    "#F97316" if rec["priority"] == "HIGH" else "#3B82F6"
                )
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {priority_color};">
                    <div style="font-weight: 700; font-size: 1.05rem;">{rec['icon']} {rec['title']}</div>
                    <div style="font-size: 0.92rem; color: #475569; margin-top: 0.3rem;">{rec['description']}</div>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: BATCH CSV TRIAGE
    # =========================================================================
    with tab2:
        st.markdown("### 📂 Batch Patient Risk Screening & Triage")
        st.caption("Upload a cohort CSV dataset to perform automated blood pressure risk classification and triage.")

        # Template Download
        sample_df = pd.DataFrame([defaults, defaults])
        csv_template = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_template,
            file_name="patient_batch_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Upload Patient Records CSV", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** patient records.")
            
            with st.spinner("Processing batch predictions..."):
                processed_df = predictor.predict_batch(batch_df)

            st.success("✅ Batch risk analysis complete!")

            # Summary metrics
            triage_c1, triage_c2, triage_c3, triage_c4 = st.columns(4)
            normal_count = (processed_df["predicted_category"] == "Normal").sum()
            elevated_count = (processed_df["predicted_category"] == "Elevated").sum()
            stage1_count = (processed_df["predicted_category"] == "Stage 1 Hypertension").sum()
            high_risk_count = (processed_df["predicted_category"].isin(["Stage 2 Hypertension", "Hypertensive Crisis"])).sum()

            triage_c1.metric("Optimal / Normal", normal_count)
            triage_c2.metric("Elevated", elevated_count)
            triage_c3.metric("Stage 1 Hypertension", stage1_count)
            triage_c4.metric("Stage 2 / Crisis", high_risk_count, delta="High Priority", delta_color="inverse")

            # Chart of Distribution
            st.markdown("#### 📊 Risk Distribution across Patient Cohort")
            cat_counts = processed_df["predicted_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            pie_fig = px.pie(
                cat_counts,
                names="Category",
                values="Count",
                color="Category",
                color_discrete_map={
                    "Normal": "#10B981",
                    "Elevated": "#EAB308",
                    "Stage 1 Hypertension": "#F97316",
                    "Stage 2 Hypertension": "#EF4444",
                    "Hypertensive Crisis": "#991B1B"
                },
                hole=0.4
            )
            st.plotly_chart(pie_fig, use_container_width=True)

            st.markdown("#### 📋 Processed Patient Records")
            st.dataframe(processed_df, use_container_width=True)

            out_csv = processed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Classified Patient Results (CSV)",
                data=out_csv,
                file_name="cardiosense_triaged_patients.csv",
                mime="text/csv"
            )

    # =========================================================================
    # TAB 3: MODEL BENCHMARKS & METRICS
    # =========================================================================
    with tab3:
        st.markdown("### 📊 Machine Learning Model Benchmarks")
        st.caption("Comparison of 5 candidate algorithms evaluated on stratified holdout test data.")

        summary = predictor.metrics_summary
        if summary and "all_benchmarks" in summary:
            bench_df = pd.DataFrame(summary["all_benchmarks"])
            bench_df = bench_df.rename(columns={
                "model_name": "Algorithm",
                "cv_macro_f1": "5-Fold CV Macro F1",
                "test_accuracy": "Accuracy",
                "test_precision_macro": "Macro Precision",
                "test_recall_macro": "Macro Recall",
                "test_f1_macro": "Macro F1 Score",
                "test_roc_auc_ovr": "ROC-AUC (OvR)"
            })
            st.dataframe(
                bench_df[["Algorithm", "Accuracy", "Macro F1 Score", "5-Fold CV Macro F1", "Macro Precision", "Macro Recall", "ROC-AUC (OvR)"]].style.format({
                    "Accuracy": "{:.3%}",
                    "Macro F1 Score": "{:.4f}",
                    "5-Fold CV Macro F1": "{:.4f}",
                    "Macro Precision": "{:.4f}",
                    "Macro Recall": "{:.4f}",
                    "ROC-AUC (OvR)": "{:.4f}"
                }).highlight_max(axis=0, color="#DEF7EC"),
                use_container_width=True
            )

            champion = summary.get("champion_model_name", "Random Forest")
            st.success(f"🏆 **Champion Model in Production:** `{champion}`")

            # Confusion Matrix Visualization
            st.markdown("#### 🎯 Confusion Matrix (Champion Model)")
            cm_data = np.array(summary["champion_metrics"]["confusion_matrix"])
            cm_fig = px.imshow(
                cm_data,
                text_auto=True,
                labels=dict(x="Predicted Risk Category", y="Actual Ground Truth Category", color="Count"),
                x=RISK_CATEGORIES,
                y=RISK_CATEGORIES,
                color_continuous_scale="Blues"
            )
            cm_fig.update_layout(height=450)
            st.plotly_chart(cm_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Clinical Reference: AHA / ACC Blood Pressure Stages")
        ref_df = pd.DataFrame([
            {"Category": "Normal", "Systolic BP": "< 120 mmHg", "Diastolic BP": "and < 80 mmHg", "Action": "Maintain heart-healthy lifestyle habits."},
            {"Category": "Elevated", "Systolic BP": "120 - 129 mmHg", "Diastolic BP": "and < 80 mmHg", "Action": "Lifestyle changes: reduce sodium, increase aerobic exercise."},
            {"Category": "Stage 1 Hypertension", "Systolic BP": "130 - 139 mmHg", "Diastolic BP": "or 80 - 89 mmHg", "Action": "Lifestyle modifications + evaluate for antihypertensive therapy."},
            {"Category": "Stage 2 Hypertension", "Systolic BP": "≥ 140 mmHg", "Diastolic BP": "or ≥ 90 mmHg", "Action": "Prescription medication + frequent clinical monitoring."},
            {"Category": "Hypertensive Crisis", "Systolic BP": "> 180 mmHg", "Diastolic BP": "and/or > 120 mmHg", "Action": "Immediate emergency medical attention."}
        ])
        st.table(ref_df)


if __name__ == "__main__":
    main()
