"""
Model evaluation, performance metrics calculation, and confusion matrix utilities.
"""

import os
import sys
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    from .data_loader import RISK_CATEGORIES
except ImportError:
    from data_loader import RISK_CATEGORIES


def evaluate_multiclass_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    target_names: List[str] = RISK_CATEGORIES
) -> Dict[str, Any]:
    """
    Computes a comprehensive evaluation dictionary for multi-class classification.
    """
    labels = list(range(len(target_names)))
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    try:
        # One-vs-Rest ROC-AUC
        roc_auc = roc_auc_score(
            y_true,
            y_proba,
            labels=labels,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        roc_auc = None

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_ovr": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


def print_evaluation_summary(metrics: Dict[str, Any], model_name: str = "Model"):
    """
    Prints a formatted summary table to console.
    """
    print(f"\n{'='*55}")
    print(f"Evaluation Metrics Summary: {model_name}")
    print(f"{'='*55}")
    print(f" Accuracy:       {metrics['accuracy']:.4f}")
    print(f" Macro Precision:{metrics['precision_macro']:.4f}")
    print(f" Macro Recall:   {metrics['recall_macro']:.4f}")
    print(f" Macro F1 Score: {metrics['f1_macro']:.4f}")
    print(f" Weighted F1:    {metrics['f1_weighted']:.4f}")
    if metrics.get('roc_auc_ovr') is not None:
        print(f" ROC-AUC (OvR):  {metrics['roc_auc_ovr']:.4f}")
    print(f"{'='*55}\n")
