"""
SHAP Explainability and Clinical Recommendation Engine for BP Risk Prediction.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px

try:
    from .data_loader import RISK_CATEGORIES
except ImportError:
    from data_loader import RISK_CATEGORIES


class BPExplainabilityEngine:
    """
    Computes patient-level feature attributions and actionable lifestyle recommendations.
    """
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.Explainer(self.model)

    def explain_instance(
        self,
        transformed_sample: np.ndarray,
        predicted_class_idx: int
    ) -> Dict[str, Any]:
        """
        Computes SHAP values for a single patient instance.
        """
        # Ensure 2D shape (1, n_features)
        if transformed_sample.ndim == 1:
            transformed_sample = transformed_sample.reshape(1, -1)

        shap_values = self.explainer(transformed_sample)
        
        # Handle multi-class SHAP outputs
        if len(shap_values.values.shape) == 3:
            # (n_samples, n_features, n_classes)
            class_shap = shap_values.values[0, :, predicted_class_idx]
            base_val = shap_values.base_values[0, predicted_class_idx]
        elif len(shap_values.values.shape) == 2:
            class_shap = shap_values.values[0, :]
            base_val = shap_values.base_values[0]
        else:
            class_shap = np.array(shap_values.values[0])
            base_val = float(np.mean(shap_values.base_values))

        feature_impacts = []
        for name, val, impact in zip(self.feature_names, transformed_sample[0], class_shap):
            feature_impacts.append({
                "feature": name,
                "value": float(val),
                "shap_impact": float(impact),
                "abs_impact": abs(float(impact))
            })

        # Sort by absolute SHAP impact descending
        feature_impacts = sorted(feature_impacts, key=lambda x: x["abs_impact"], reverse=True)

        return {
            "predicted_class": RISK_CATEGORIES[predicted_class_idx],
            "base_value": float(base_val),
            "top_features": feature_impacts[:10],
            "all_features": feature_impacts
        }

    def generate_clinical_recommendations(
        self,
        raw_patient_dict: Dict[str, Any],
        predicted_category: str
    ) -> List[Dict[str, str]]:
        """
        Generates evidence-based clinical and lifestyle recommendations tailored to patient risks.
        """
        recs = []

        # 1. Immediate Critical Alerts
        if predicted_category == "Hypertensive Crisis":
            recs.append({
                "priority": "CRITICAL",
                "icon": "🚨",
                "title": "Emergency Medical Evaluation Required",
                "description": "Blood pressure exceeds 180 mmHg systolic or 120 mmHg diastolic. Please seek emergency medical care immediately."
            })

        if predicted_category in ["Stage 2 Hypertension", "Stage 1 Hypertension"]:
            recs.append({
                "priority": "HIGH",
                "icon": "🩺",
                "title": "Physician Consultation & BP Monitoring",
                "description": "Schedule a clinical evaluation with your healthcare provider for diagnostic confirmation and tailored pharmacotherapy evaluation."
            })

        # 2. Dietary Sodium
        sodium = raw_patient_dict.get("daily_sodium_mg", 3000)
        if sodium > 2300:
            recs.append({
                "priority": "HIGH" if sodium > 3500 else "MEDIUM",
                "icon": "🧂",
                "title": "Adopt AHA Low-Sodium / DASH Diet",
                "description": f"Current intake is ~{int(sodium)} mg/day. AHA recommends reducing sodium intake to under 1,500–2,300 mg/day, which can reduce systolic BP by 5–8 mmHg."
            })

        # 3. BMI & Weight Management
        bmi = raw_patient_dict.get("bmi", 25)
        if bmi >= 25.0:
            status = "Obese (BMI ≥ 30)" if bmi >= 30 else "Overweight (BMI 25-29.9)"
            recs.append({
                "priority": "HIGH" if bmi >= 30 else "MEDIUM",
                "icon": "⚖️",
                "title": f"Weight Management ({status})",
                "description": f"Your BMI is {bmi:.1f}. Each 1 kg reduction in body weight is associated with an approximate 1 mmHg drop in systolic blood pressure."
            })

        # 4. Physical Activity
        activity = raw_patient_dict.get("physical_activity_hours", 2.0)
        if activity < 2.5:
            recs.append({
                "priority": "MEDIUM",
                "icon": "🏃",
                "title": "Increase Aerobic Physical Activity",
                "description": f"Currently at {activity:.1f} hrs/week. Target at least 150 minutes (2.5 hours) of moderate-intensity aerobic exercise (e.g. brisk walking, cycling) weekly."
            })

        # 5. Smoking Status
        smoking = raw_patient_dict.get("smoking_status", "Non-Smoker")
        if smoking == "Current Smoker":
            recs.append({
                "priority": "HIGH",
                "icon": "🚭",
                "title": "Smoking Cessation Support",
                "description": "Smoking causes acute spikes in blood pressure and accelerates arterial atherosclerosis. Quitting dramatically reduces cardiovascular risk within 1 year."
            })

        # 6. Stress & Sleep
        stress = raw_patient_dict.get("stress_level", 5)
        sleep = raw_patient_dict.get("sleep_hours", 7.0)
        if stress >= 7:
            recs.append({
                "priority": "MEDIUM",
                "icon": "🧘",
                "title": "Stress Reduction & Mindfulness",
                "description": f"Elevated stress score ({stress}/10) activates sympathetic vasoconstriction. Consider mindfulness meditation, deep breathing, or yoga."
            })
        if sleep < 6.0:
            recs.append({
                "priority": "MEDIUM",
                "icon": "🌙",
                "title": "Optimize Sleep Duration",
                "description": f"Current sleep ({sleep:.1f} hrs) is below the recommended 7–9 hours. Chronic sleep deprivation increases nocturnal autonomic tone and hypertension."
            })

        # 7. Lipid & Metabolic Health
        chol = raw_patient_dict.get("cholesterol_total", 180)
        fbs = raw_patient_dict.get("fasting_blood_sugar", 90)
        if chol > 200 or fbs > 100:
            recs.append({
                "priority": "MEDIUM",
                "icon": "🩸",
                "title": "Metabolic & Lipid Panel Follow-up",
                "description": "Fasting blood sugar or cholesterol levels indicate elevated metabolic cardiovascular risk. Consult your physician for comprehensive lipid management."
            })

        if not recs:
            recs.append({
                "priority": "LOW",
                "icon": "✅",
                "title": "Maintain Healthy Lifestyle",
                "description": "All clinical biomarkers appear within optimal ranges. Continue balanced nutrition, regular exercise, and annual preventive health check-ups."
            })

        return recs


def plot_shap_waterfall_interactive(top_features: List[Dict[str, Any]], base_value: float, predicted_category: str):
    """
    Creates an interactive Plotly horizontal bar chart showing top SHAP feature contributions.
    """
    features = [item["feature"] for item in top_features][::-1]
    impacts = [item["shap_impact"] for item in top_features][::-1]
    colors = ["#EF4444" if val > 0 else "#10B981" for val in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="#1E293B")),
        hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>SHAP Feature Attribution</b> for Risk: <i>{predicted_category}</i>",
        xaxis_title="SHAP Value (Impact on Prediction Log-Odds)",
        yaxis_title="Feature Name",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=50, b=30)
    )

    return fig
