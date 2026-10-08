"""
Data preprocessing, feature engineering, and pipeline transformations for BP Risk Prediction.
"""

import os
import sys
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Ensure module alias for unpickling compatibility
sys.modules["preprocessor"] = sys.modules[__name__]

try:
    from .data_loader import (
        RISK_CATEGORIES,
        RISK_CATEGORY_MAP,
        INVERSE_RISK_CATEGORY_MAP,
        TARGET_COLUMN
    )
except ImportError:
    from data_loader import (
        RISK_CATEGORIES,
        RISK_CATEGORY_MAP,
        INVERSE_RISK_CATEGORY_MAP,
        TARGET_COLUMN
    )


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer physiological and clinical cardiovascular indices.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. Pulse Pressure (SBP - DBP) - marker of arterial stiffness
        if "systolic_bp" in X_out.columns and "diastolic_bp" in X_out.columns:
            X_out["pulse_pressure"] = X_out["systolic_bp"] - X_out["diastolic_bp"]
            # 2. Mean Arterial Pressure (MAP)
            X_out["mean_arterial_pressure"] = X_out["diastolic_bp"] + (X_out["pulse_pressure"] / 3.0)
        else:
            X_out["pulse_pressure"] = 40.0
            X_out["mean_arterial_pressure"] = 93.3

        # 3. Cholesterol Ratio (Total / HDL)
        if "cholesterol_total" in X_out.columns and "cholesterol_hdl" in X_out.columns:
            hdl_safe = np.maximum(X_out["cholesterol_hdl"].values, 1.0)
            X_out["cholesterol_ratio"] = X_out["cholesterol_total"].values / hdl_safe
        else:
            X_out["cholesterol_ratio"] = 3.5

        # 4. Metabolic Composite Index
        bmi_val = X_out["bmi"] if "bmi" in X_out.columns else 24.0
        fbs_val = X_out["fasting_blood_sugar"] if "fasting_blood_sugar" in X_out.columns else 90.0
        tg_val = X_out["triglycerides"] if "triglycerides" in X_out.columns else 120.0
        X_out["metabolic_risk_index"] = (bmi_val / 25.0) + (fbs_val / 100.0) + (tg_val / 150.0)

        # 5. Lifestyle Stress & Strain Index
        stress = X_out["stress_level"] if "stress_level" in X_out.columns else 5.0
        sodium = X_out["daily_sodium_mg"] if "daily_sodium_mg" in X_out.columns else 3000.0
        activity = X_out["physical_activity_hours"] if "physical_activity_hours" in X_out.columns else 2.5
        sleep = X_out["sleep_hours"] if "sleep_hours" in X_out.columns else 7.0
        X_out["lifestyle_strain_score"] = (
            (stress / 10.0) * 1.5 
            + (sodium / 2300.0) 
            - (activity / 4.0) 
            - ((sleep - 7.0) / 4.0)
        )

        return X_out


def create_preprocessor_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> Pipeline:
    """
    Creates full preprocessing and scaling sklearn pipeline.
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    col_transformer = ColumnTransformer(transformers=[
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ], remainder="drop")

    full_pipeline = Pipeline(steps=[
        ("feature_engineer", ClinicalFeatureEngineer()),
        ("preprocessor", col_transformer)
    ])

    return full_pipeline


def get_feature_names_after_preprocessing(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """
    Extracts column names output by the preprocessor pipeline.
    """
    col_transformer = pipeline.named_steps["preprocessor"]
    cat_encoder = col_transformer.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    return numeric_features + cat_names


def prepare_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Splits dataframe into X, y and defines feature lists.
    """
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].map(RISK_CATEGORY_MAP)
    else:
        y = None

    # Base + engineered numeric features
    numeric_features = [
        "age",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "bmi",
        "cholesterol_total",
        "cholesterol_hdl",
        "cholesterol_ldl",
        "triglycerides",
        "fasting_blood_sugar",
        "physical_activity_hours",
        "daily_sodium_mg",
        "alcohol_drinks_per_week",
        "stress_level",
        "sleep_hours",
        "family_history_hypertension",
        "has_diabetes",
        "has_ckd",
        "pulse_pressure",
        "mean_arterial_pressure",
        "cholesterol_ratio",
        "metabolic_risk_index",
        "lifestyle_strain_score"
    ]

    categorical_features = ["gender", "smoking_status"]

    return X, y, numeric_features, categorical_features
