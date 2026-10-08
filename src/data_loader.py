"""
Data loading and synthetic clinical dataset generator for Blood Pressure Risk Prediction.
Complies with AHA/ACC (American Heart Association / American College of Cardiology) clinical guidelines.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Clinical Risk Categories defined by AHA/ACC Guidelines
RISK_CATEGORIES = [
    "Normal",
    "Elevated",
    "Stage 1 Hypertension",
    "Stage 2 Hypertension",
    "Hypertensive Crisis"
]

RISK_CATEGORY_MAP = {cat: idx for idx, cat in enumerate(RISK_CATEGORIES)}
INVERSE_RISK_CATEGORY_MAP = {idx: cat for idx, cat in enumerate(RISK_CATEGORIES)}

FEATURE_COLUMNS = [
    "age",
    "gender",
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
    "smoking_status",
    "alcohol_drinks_per_week",
    "stress_level",
    "sleep_hours",
    "family_history_hypertension",
    "has_diabetes",
    "has_ckd"
]

TARGET_COLUMN = "bp_risk_category"


def assign_aha_bp_category(systolic: float, diastolic: float) -> str:
    """
    Classify blood pressure reading into clinical risk category based on AHA/ACC 2017 Guidelines.
    """
    if systolic >= 180 or diastolic >= 120:
        return "Hypertensive Crisis"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension"
    elif (130 <= systolic < 140) or (80 <= diastolic < 90):
        return "Stage 1 Hypertension"
    elif (120 <= systolic < 130) and diastolic < 80:
        return "Elevated"
    else:
        return "Normal"


def generate_synthetic_clinical_data(
    n_samples: int = 3500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, clinically calibrated dataset modeling physiological and lifestyle variables.
    Ensures representative sampling across all 5 AHA blood pressure risk categories.
    """
    rng = np.random.default_rng(random_state)

    # 1. Demographics
    age = rng.integers(18, 85, size=n_samples)
    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.50, 0.50])

    # 2. Lifestyle & Habits
    smoking_status = rng.choice(["Non-Smoker", "Former Smoker", "Current Smoker"], size=n_samples, p=[0.50, 0.25, 0.25])
    smoking_score = np.where(smoking_status == "Current Smoker", 1.0, np.where(smoking_status == "Former Smoker", 0.3, 0.0))

    physical_activity_hours = np.clip(rng.gamma(shape=2.5, scale=1.5, size=n_samples), 0, 20)
    daily_sodium_mg = np.clip(rng.normal(loc=3200, scale=900, size=n_samples), 800, 6500)
    alcohol_drinks = np.clip(rng.exponential(scale=3.5, size=n_samples), 0, 30)
    stress_level = rng.integers(1, 11, size=n_samples) # 1 to 10 scale
    sleep_hours = np.clip(rng.normal(loc=6.8, scale=1.3, size=n_samples), 3.5, 11.0)

    # 3. Medical History
    family_history = rng.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    
    # 4. Biometrics (BMI correlated with activity, age, diet)
    bmi_base = 21.0 + (age * 0.06) - (physical_activity_hours * 0.35) + (daily_sodium_mg / 3500.0) + rng.normal(0, 3.2, n_samples)
    bmi = np.clip(bmi_base, 16.0, 50.0)

    # 5. Comorbidities
    diabetes_prob = 1 / (1 + np.exp(-( -4.0 + 0.04 * age + 0.08 * bmi + 0.5 * family_history )))
    has_diabetes = (rng.uniform(0, 1, n_samples) < diabetes_prob).astype(int)

    ckd_prob = 1 / (1 + np.exp(-( -5.5 + 0.05 * age + 0.05 * bmi + 0.8 * has_diabetes )))
    has_ckd = (rng.uniform(0, 1, n_samples) < ckd_prob).astype(int)

    # 6. Biochemical Markers
    fbs_base = 85.0 + (has_diabetes * 45.0) + (bmi * 0.8) + rng.normal(0, 12, n_samples)
    fasting_blood_sugar = np.clip(fbs_base, 65.0, 300.0)

    chol_total = np.clip(160 + (age * 0.6) + (bmi * 1.5) + (smoking_score * 15) + rng.normal(0, 25, n_samples), 110, 360)
    chol_hdl = np.clip(60 - (bmi * 0.35) + (physical_activity_hours * 0.8) - (smoking_score * 8) + rng.normal(0, 8, n_samples), 22, 95)
    chol_ldl = np.clip(chol_total - chol_hdl - 30 + rng.normal(0, 12, n_samples), 45, 250)
    triglycerides = np.clip(90 + (bmi * 2.5) + (alcohol_drinks * 3.0) + (has_diabetes * 35) + rng.normal(0, 35, n_samples), 40, 500)

    # 7. Resting Heart Rate
    hr_base = 72 + (stress_level * 1.2) - (physical_activity_hours * 1.0) + (smoking_score * 5) + (alcohol_drinks * 0.4) + rng.normal(0, 7, n_samples)
    heart_rate = np.clip(hr_base, 45, 125)

    # 8. Physiological Blood Pressure Generation
    # Create diverse sub-cohorts (Optimal, Elevated, Stage 1, Stage 2, Hypertensive crisis)
    cohort_assignment = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.24, 0.18, 0.28, 0.22, 0.08])

    sbp_list = []
    dbp_list = []

    for i in range(n_samples):
        cohort = cohort_assignment[i]
        
        if cohort == 0:  # Normal target
            sbp = rng.normal(110, 5)
            dbp = rng.normal(72, 4)
        elif cohort == 1:  # Elevated target (SBP 120-129, DBP < 80)
            sbp = rng.uniform(121, 128)
            dbp = rng.uniform(68, 78)
        elif cohort == 2:  # Stage 1 (SBP 130-139 or DBP 80-89)
            sbp = rng.uniform(131, 138)
            dbp = rng.uniform(81, 88)
        elif cohort == 3:  # Stage 2 (SBP 140-175 or DBP 90-115)
            sbp = rng.uniform(142, 172)
            dbp = rng.uniform(91, 112)
        else:  # Hypertensive Crisis (SBP >= 180 or DBP >= 120)
            sbp = rng.uniform(182, 215)
            dbp = rng.uniform(115, 130)

        # Apply minor physiological perturbations correlated with patient covariates
        sbp_adjusted = sbp + (age[i] * 0.05) + (bmi[i] * 0.1) + ((daily_sodium_mg[i] - 2500)/2000.0) + rng.normal(0, 2)
        dbp_adjusted = dbp + (age[i] * 0.02) + (bmi[i] * 0.05) + rng.normal(0, 1.5)

        # Enforce realistic bounds
        sbp_final = np.clip(np.round(sbp_adjusted), 85, 230)
        dbp_final = np.clip(np.round(dbp_adjusted), 50, 140)
        dbp_final = np.minimum(dbp_final, sbp_final - 18) # SBP strictly higher than DBP

        sbp_list.append(sbp_final)
        dbp_list.append(dbp_final)

    systolic_bp = np.array(sbp_list)
    diastolic_bp = np.array(dbp_list)

    # 9. Assign Target Label based on AHA Ground Truth
    categories = [assign_aha_bp_category(sbp, dbp) for sbp, dbp in zip(systolic_bp, diastolic_bp)]

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "systolic_bp": systolic_bp.astype(float),
        "diastolic_bp": diastolic_bp.astype(float),
        "heart_rate": np.round(heart_rate, 1),
        "bmi": np.round(bmi, 1),
        "cholesterol_total": np.round(chol_total, 1),
        "cholesterol_hdl": np.round(chol_hdl, 1),
        "cholesterol_ldl": np.round(chol_ldl, 1),
        "triglycerides": np.round(triglycerides, 1),
        "fasting_blood_sugar": np.round(fasting_blood_sugar, 1),
        "physical_activity_hours": np.round(physical_activity_hours, 1),
        "daily_sodium_mg": np.round(daily_sodium_mg, 0),
        "smoking_status": smoking_status,
        "alcohol_drinks_per_week": np.round(alcohol_drinks, 1),
        "stress_level": stress_level,
        "sleep_hours": np.round(sleep_hours, 1),
        "family_history_hypertension": family_history,
        "has_diabetes": has_diabetes,
        "has_ckd": has_ckd,
        TARGET_COLUMN: categories
    })

    return df


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads dataset from file_path if provided, else generates synthetic clinical data.
    """
    if file_path:
        df = pd.read_csv(file_path)
        return df
    return generate_synthetic_clinical_data(n_samples=3500)
