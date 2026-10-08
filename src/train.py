"""
Training, benchmarking, cross-validation, and serialization pipeline for BP Risk Prediction.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# Ensure standard UTF-8 stdout if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path if run directly
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from .data_loader import load_dataset, RISK_CATEGORIES
    from .preprocessor import (
        create_preprocessor_pipeline,
        prepare_data,
        get_feature_names_after_preprocessing
    )
    from .evaluate import evaluate_multiclass_model, print_evaluation_summary
except ImportError:
    from data_loader import load_dataset, RISK_CATEGORIES
    from preprocessor import (
        create_preprocessor_pipeline,
        prepare_data,
        get_feature_names_after_preprocessing
    )
    from evaluate import evaluate_multiclass_model, print_evaluation_summary


MODELS_DIR = os.path.join(os.path.dirname(SRC_DIR), "models")
DATA_DIR = os.path.join(os.path.dirname(SRC_DIR), "data")


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns candidate classification algorithms for benchmarking.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
            class_weight="balanced"
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            eval_metric="mlogloss"
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            num_leaves=31,
            random_state=random_state,
            class_weight="balanced",
            verbose=-1
        ),
        "Support Vector Machine": SVC(
            C=2.0,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=random_state
        )
    }


def train_and_benchmark_models(
    data_path: str = None,
    n_samples: int = 3500,
    random_state: int = 42,
    output_dir: str = MODELS_DIR
) -> Dict[str, Any]:
    """
    Runs full end-to-end training and evaluation benchmarking across multiple algorithms.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1/5] Loading / Generating clinical dataset...")
    df = load_dataset(data_path)
    
    # Save raw generated dataset for reference
    raw_data_path = os.path.join(DATA_DIR, "clinical_bp_dataset.csv")
    df.to_csv(raw_data_path, index=False)
    print(f"[+] Dataset saved to {raw_data_path} ({len(df)} records)")

    X, y, numeric_features, categorical_features = prepare_data(df)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    print(f"[2/5] Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Fit Preprocessing Pipeline
    preprocessor = create_preprocessor_pipeline(numeric_features, categorical_features)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names_after_preprocessing(
        preprocessor, numeric_features, categorical_features
    )

    # Benchmark candidate models
    candidate_models = get_candidate_models(random_state)
    benchmarks = []
    trained_models = {}
    evaluation_reports = {}

    best_score = -1.0
    best_model_name = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print("[3/5] Benchmarking candidate algorithms...")
    for name, model in candidate_models.items():
        print(f" -> Training & Cross-Validating: {name}...")
        
        # 5-Fold Cross-Validation Macro F1
        cv_scores = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring="f1_macro")
        mean_cv_f1 = float(np.mean(cv_scores))
        std_cv_f1 = float(np.std(cv_scores))

        # Fit on full training set
        model.fit(X_train_trans, y_train)
        trained_models[name] = model

        # Test set evaluation
        y_pred = model.predict(X_test_trans)
        y_proba = model.predict_proba(X_test_trans)
        eval_metrics = evaluate_multiclass_model(y_test, y_pred, y_proba)
        evaluation_reports[name] = eval_metrics

        print(f"    * 5-Fold CV Macro F1: {mean_cv_f1:.4f} (+/- {std_cv_f1:.4f})")
        print(f"    * Test Accuracy:      {eval_metrics['accuracy']:.4f}")
        print(f"    * Test Macro F1:      {eval_metrics['f1_macro']:.4f}")

        benchmarks.append({
            "model_name": name,
            "cv_macro_f1": mean_cv_f1,
            "cv_std": std_cv_f1,
            "test_accuracy": eval_metrics["accuracy"],
            "test_precision_macro": eval_metrics["precision_macro"],
            "test_recall_macro": eval_metrics["recall_macro"],
            "test_f1_macro": eval_metrics["f1_macro"],
            "test_f1_weighted": eval_metrics["f1_weighted"],
            "test_roc_auc_ovr": eval_metrics.get("roc_auc_ovr")
        })

        if eval_metrics["f1_macro"] > best_score:
            best_score = eval_metrics["f1_macro"]
            best_model_name = name

    # Convert benchmarks to DataFrame
    benchmark_df = pd.DataFrame(benchmarks).sort_values(by="test_f1_macro", ascending=False)
    benchmark_csv_path = os.path.join(output_dir, "model_comparison.csv")
    benchmark_df.to_csv(benchmark_csv_path, index=False)

    print(f"\n[4/5] Champion Model Selected: {best_model_name} (Test Macro F1: {best_score:.4f})")

    # Serialize Best Model and Pipeline
    best_model = trained_models[best_model_name]
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    feature_names_path = os.path.join(output_dir, "feature_names.json")
    metrics_summary_path = os.path.join(output_dir, "metrics_summary.json")

    joblib.dump(best_model, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)

    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    summary_data = {
        "champion_model_name": best_model_name,
        "champion_metrics": evaluation_reports[best_model_name],
        "all_benchmarks": benchmarks,
        "dataset_size": len(df),
        "test_size": len(X_test),
        "classes": RISK_CATEGORIES
    }

    with open(metrics_summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[5/5] All models and artifacts successfully saved to {output_dir}")

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BP Risk Prediction Models")
    parser.add_argument("--data-path", type=str, default=None, help="Path to custom CSV dataset")
    parser.add_argument("--samples", type=int, default=3500, help="Number of synthetic samples if no data path")
    args = parser.parse_args()

    train_and_benchmark_models(data_path=args.data_path, n_samples=args.samples)
