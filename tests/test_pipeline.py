"""
Automated unit tests for Blood Pressure Risk Prediction pipeline, preprocessor, and predictor.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add workspace to sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
WORKSPACE_DIR = os.path.dirname(PROJECT_DIR)
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from bp_risk_prediction.src.data_loader import (
    assign_aha_bp_category,
    generate_synthetic_clinical_data,
    RISK_CATEGORIES,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from bp_risk_prediction.src.preprocessor import (
    ClinicalFeatureEngineer,
    create_preprocessor_pipeline,
    prepare_data,
    get_feature_names_after_preprocessing
)
from bp_risk_prediction.src.evaluate import evaluate_multiclass_model
from bp_risk_prediction.src.predictor import BPRiskPredictor


def test_aha_bp_category_rules():
    assert assign_aha_bp_category(115, 75) == "Normal"
    assert assign_aha_bp_category(125, 78) == "Elevated"
    assert assign_aha_bp_category(132, 78) == "Stage 1 Hypertension"
    assert assign_aha_bp_category(125, 84) == "Stage 1 Hypertension"
    assert assign_aha_bp_category(145, 88) == "Stage 2 Hypertension"
    assert assign_aha_bp_category(130, 95) == "Stage 2 Hypertension"
    assert assign_aha_bp_category(185, 90) == "Hypertensive Crisis"
    assert assign_aha_bp_category(140, 125) == "Hypertensive Crisis"


def test_synthetic_data_generator():
    df = generate_synthetic_clinical_data(n_samples=100, random_state=42)
    assert len(df) == 100
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing column {col}"
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].isin(RISK_CATEGORIES).all()


def test_clinical_feature_engineer():
    df = pd.DataFrame([{
        "systolic_bp": 140.0,
        "diastolic_bp": 90.0,
        "cholesterol_total": 210.0,
        "cholesterol_hdl": 50.0,
        "bmi": 28.0,
        "fasting_blood_sugar": 105.0,
        "triglycerides": 160.0,
        "stress_level": 6,
        "daily_sodium_mg": 3200.0,
        "physical_activity_hours": 2.0,
        "sleep_hours": 6.5
    }])

    fe = ClinicalFeatureEngineer()
    df_trans = fe.transform(df)

    assert "pulse_pressure" in df_trans.columns
    assert df_trans["pulse_pressure"].iloc[0] == 50.0
    assert "mean_arterial_pressure" in df_trans.columns
    assert abs(df_trans["mean_arterial_pressure"].iloc[0] - (90 + 50/3)) < 1e-4
    assert "cholesterol_ratio" in df_trans.columns
    assert abs(df_trans["cholesterol_ratio"].iloc[0] - 4.2) < 1e-4
    assert "metabolic_risk_index" in df_trans.columns
    assert "lifestyle_strain_score" in df_trans.columns


def test_preprocessor_pipeline_transformation():
    df = generate_synthetic_clinical_data(n_samples=50, random_state=42)
    X, y, num_cols, cat_cols = prepare_data(df)

    pipeline = create_preprocessor_pipeline(num_cols, cat_cols)
    X_trans = pipeline.fit_transform(X)

    feature_names = get_feature_names_after_preprocessing(pipeline, num_cols, cat_cols)

    assert X_trans.shape[0] == 50
    assert X_trans.shape[1] == len(feature_names)
    assert not np.isnan(X_trans).any()


def test_evaluation_metric_computation():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    y_proba = np.eye(5)

    metrics = evaluate_multiclass_model(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert len(metrics["confusion_matrix"]) == 5


def test_predictor_single_and_batch_inference():
    models_dir = os.path.join(PROJECT_DIR, "models")
    if os.path.exists(os.path.join(models_dir, "best_model.joblib")):
        predictor = BPRiskPredictor(models_dir=models_dir)

        sample_patient = {
            "age": 45,
            "gender": "Male",
            "systolic_bp": 135.0,
            "diastolic_bp": 85.0,
            "heart_rate": 75.0,
            "bmi": 26.5,
            "cholesterol_total": 190.0,
            "cholesterol_hdl": 48.0,
            "cholesterol_ldl": 120.0,
            "triglycerides": 140.0,
            "fasting_blood_sugar": 95.0,
            "physical_activity_hours": 3.0,
            "daily_sodium_mg": 2800.0,
            "smoking_status": "Non-Smoker",
            "alcohol_drinks_per_week": 2.0,
            "stress_level": 5,
            "sleep_hours": 7.0,
            "family_history_hypertension": 0,
            "has_diabetes": 0,
            "has_ckd": 0
        }

        result = predictor.predict_single(sample_patient)
        assert "predicted_category" in result
        assert result["predicted_category"] in RISK_CATEGORIES
        assert "risk_score_100" in result
        assert 0.0 <= result["risk_score_100"] <= 100.0
        assert "probabilities" in result
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

        # Batch test
        batch_df = pd.DataFrame([sample_patient, sample_patient])
        batch_out = predictor.predict_batch(batch_df)
        assert len(batch_out) == 2
        assert "predicted_category" in batch_out.columns
        assert "risk_score_100" in batch_out.columns
