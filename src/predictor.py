"""
Inference engine and real-time predictor for Blood Pressure Risk Assessment.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure module aliases so unpickling works seamlessly in all environments
try:
    from . import preprocessor as _prep_mod
    sys.modules["preprocessor"] = _prep_mod
    sys.modules["bp_risk_prediction.src.preprocessor"] = _prep_mod
except ImportError:
    try:
        import bp_risk_prediction.src.preprocessor as _prep_mod
        sys.modules["preprocessor"] = _prep_mod
    except ImportError:
        import preprocessor as _prep_mod
        sys.modules["preprocessor"] = _prep_mod

try:
    from .data_loader import (
        RISK_CATEGORIES,
        INVERSE_RISK_CATEGORY_MAP,
        FEATURE_COLUMNS
    )
    from .explainability import BPExplainabilityEngine
except ImportError:
    from data_loader import (
        RISK_CATEGORIES,
        INVERSE_RISK_CATEGORY_MAP,
        FEATURE_COLUMNS
    )
    from explainability import BPExplainabilityEngine


DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class BPRiskPredictor:
    """
    Production-ready predictor providing inference, risk confidence, SHAP attribution, and clinical recommendations.
    """
    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.feature_names_path = os.path.join(models_dir, "feature_names.json")
        self.metrics_path = os.path.join(models_dir, "metrics_summary.json")

        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Model artifacts not found in {models_dir}. Please run train.py first to train models."
            )

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        with open(self.feature_names_path, "r") as f:
            self.feature_names = json.load(f)

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r") as f:
                self.metrics_summary = json.load(f)
        else:
            self.metrics_summary = {}

        self.explainer = BPExplainabilityEngine(self.model, self.feature_names)

    def predict_single(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs complete inference, probability estimation, SHAP attribution, and recommendations for a single patient.
        """
        # Convert to DataFrame
        df_input = pd.DataFrame([patient_dict])
        
        # Transform features
        X_trans = self.preprocessor.transform(df_input)

        # Model prediction
        pred_idx = int(self.model.predict(X_trans)[0])
        probabilities = self.model.predict_proba(X_trans)[0]

        pred_category = RISK_CATEGORIES[pred_idx]
        confidence = float(probabilities[pred_idx])

        # Probability distribution dict
        prob_dict = {
            RISK_CATEGORIES[i]: float(probabilities[i])
            for i in range(len(RISK_CATEGORIES))
        }

        # SHAP Explainability
        try:
            explanation = self.explainer.explain_instance(X_trans, pred_idx)
        except Exception as e:
            explanation = {"error": str(e), "top_features": []}

        # Tailored Clinical Recommendations
        recommendations = self.explainer.generate_clinical_recommendations(
            patient_dict, pred_category
        )

        # Risk severity score (0 to 100)
        severity_weights = [10, 35, 60, 85, 100]
        risk_score = float(np.dot(probabilities, severity_weights))

        return {
            "predicted_category": pred_category,
            "category_index": pred_idx,
            "confidence": confidence,
            "risk_score_100": round(risk_score, 1),
            "probabilities": prob_dict,
            "explanation": explanation,
            "recommendations": recommendations,
            "patient_data": patient_dict
        }

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Performs batch inference on an input DataFrame and appends prediction columns.
        """
        df_out = df_input.copy()
        X_trans = self.preprocessor.transform(df_out)

        preds = self.model.predict(X_trans)
        probas = self.model.predict_proba(X_trans)

        severity_weights = np.array([10, 35, 60, 85, 100])
        risk_scores = np.dot(probas, severity_weights)

        df_out["predicted_category"] = [RISK_CATEGORIES[i] for i in preds]
        df_out["prediction_confidence"] = np.max(probas, axis=1).round(3)
        df_out["risk_score_100"] = risk_scores.round(1)

        for idx, cat in enumerate(RISK_CATEGORIES):
            col_name = f"prob_{cat.lower().replace(' ', '_')}"
            df_out[col_name] = probas[:, idx].round(3)

        return df_out
