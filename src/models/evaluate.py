"""
Model evaluation, metrics computation, and report generation.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.features.build_features import INT_TO_LABEL
from src.utils.config import setup_logger

logger = setup_logger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    save_artifacts: bool = True,
    output_dir: str = "reports/",
) -> Dict[str, Any]:
    """
    Computes classification metrics, logs results, and generates visualizations.
    """
    labels = [0, 1, 2]
    target_names = [INT_TO_LABEL[i] for i in labels]

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    clf_report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_score_macro": f1_macro,
        "confusion_matrix": cm,
        "classification_report": clf_report,
    }

    logger.info(
        "[%s] Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1-Score: %.4f",
        model_name,
        acc,
        prec_macro,
        rec_macro,
        f1_macro,
    )

    if save_artifacts:
        metrics_path = Path(output_dir) / "metrics" / f"{model_name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Plot confusion matrix
        plot_confusion_matrix(
            y_true,
            y_pred,
            model_name=model_name,
            output_path=str(Path(output_dir) / "figures" / f"{model_name}_confusion_matrix.png"),
        )

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """
    Plots and saves a heatmap of the confusion matrix.
    """
    labels = [0, 1, 2]
    class_names = [INT_TO_LABEL[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", output_path)
