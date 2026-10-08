# 🩺 CardioSense: Blood Pressure Risk Prediction Using Machine Learning

An end-to-end, clinically grounded Machine Learning system and interactive clinical decision support application that predicts **Blood Pressure Risk Categories** based on the **2017 American Heart Association / American College of Cardiology (AHA/ACC)** guidelines.

---

## 🎯 Problem Statement & Clinical Relevance

High blood pressure (hypertension) is a leading preventable risk factor for cardiovascular disease, stroke, myocardial infarction, and chronic kidney disease. Early risk stratification and patient awareness enable timely lifestyle and pharmacological interventions.

This project delivers:
1. **Multi-Class Risk Stratification**: Classifies patients into 5 AHA categories (*Normal*, *Elevated*, *Stage 1 Hypertension*, *Stage 2 Hypertension*, *Hypertensive Crisis*).
2. **Comprehensive Feature Engineering**: Integrates demographics, vitals, biochemical lipid/glucose markers, lifestyle habits, and derived physiological metrics (Mean Arterial Pressure, Pulse Pressure, Cholesterol Ratio).
3. **Multi-Model Machine Learning Benchmark**: Trains and validates 5 candidate algorithms (*Random Forest*, *XGBoost*, *LightGBM*, *SVM*, *Logistic Regression*) with 5-fold cross-validation.
4. **Explainable AI (SHAP)**: Provides patient-level waterfall attribution explaining *why* the model made each risk prediction.
5. **Interactive Streamlit Web Dashboard**: Features single-patient risk assessment, real-time risk severity gauge, SHAP visualizer, personalized clinical recommendations, batch CSV screening, and model performance metrics.

---

## 🏥 Clinical Risk Categorization (AHA / ACC Guidelines)

| Risk Category | Systolic BP (mmHg) | Diastolic BP (mmHg) | Clinical Recommended Action |
| :--- | :--- | :--- | :--- |
| 🟢 **Normal (Optimal)** | $< 120$ | and $< 80$ | Maintain heart-healthy nutrition and active lifestyle. |
| 🟡 **Elevated** | $120 - 129$ | and $< 80$ | Lifestyle modifications: sodium reduction, aerobic activity. |
| 🟠 **Stage 1 Hypertension** | $130 - 139$ | or $80 - 89$ | Lifestyle changes + clinical evaluation for antihypertensive therapy. |
| 🔴 **Stage 2 Hypertension** | $\ge 140$ | or $\ge 90$ | Antihypertensive pharmacotherapy + regular BP monitoring. |
| 🚨 **Hypertensive Crisis** | $> 180$ | and/or $> 120$ | Immediate emergency medical evaluation required. |

---

## 🏗️ System Architecture

```
bp_risk_prediction/
├── data/
│   └── clinical_bp_dataset.csv     # Calibrated clinical cohort dataset
├── models/
│   ├── best_model.joblib           # Serialized champion ML model
│   ├── preprocessor.joblib         # Fitted sklearn preprocessing pipeline
│   ├── feature_names.json          # Engineered feature name list
│   ├── metrics_summary.json        # Benchmark performance statistics
│   └── model_comparison.csv        # Multi-model evaluation table
├── src/
│   ├── data_loader.py              # Clinical dataset generator & AHA ground truth assigner
│   ├── preprocessor.py             # Feature engineering, scaling & ColumnTransformer
│   ├── train.py                    # Multi-model benchmarking, 5-fold CV & artifact export
│   ├── evaluate.py                 # Multi-class metrics (Accuracy, F1, ROC-AUC, CM)
│   ├── explainability.py           # SHAP TreeExplainer & clinical recommendation engine
│   └── predictor.py                # High-level inference engine for single & batch predictions
├── app/
│   └── streamlit_app.py            # Modern Streamlit Web Application
├── tests/
│   └── test_pipeline.py            # Automated unit tests
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ⚡ Quick Start Guide

### 1. Install Dependencies
```bash
python -m pip install -r bp_risk_prediction/requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest bp_risk_prediction/tests/test_pipeline.py
```

### 3. Train & Benchmark ML Models
```bash
python bp_risk_prediction/src/train.py --samples 3500
```

### 4. Launch Interactive Web App
```bash
streamlit run bp_risk_prediction/app/streamlit_app.py
```

---

## 🔬 Model Benchmarking & Evaluation

The system evaluates 5 machine learning models using **Stratified 5-Fold Cross-Validation** and a holdout test set:

- **Random Forest Classifier**: Multi-tree ensemble with balanced class weighting.
- **XGBoost Classifier**: Extreme gradient boosted decision trees.
- **LightGBM Classifier**: Leaf-wise gradient boosting optimized for speed and tabular accuracy.
- **Support Vector Machine (RBF Kernel)**: Non-linear hyper-plane classification with calibrated probabilities.
- **Multinomial Logistic Regression**: Linear baseline with L2 regularization.

---

## 🔍 Explainable AI (SHAP) & Decision Support

CardioSense utilizes **SHAP (SHapley Additive exPlanations)** to ensure healthcare transparency:
- Evaluates individual patient feature contributions against baseline risk.
- Highlights primary risk drivers (e.g. elevated daily sodium intake, BMI, pulse pressure, family history).
- Delivers tailored clinical guidance (AHA DASH diet tips, weight reduction impact, physical activity goals).
