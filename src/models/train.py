"""
Model training, cross-validation, comparison, and serialization module.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data.make_dataset import load_raw_data, validate_and_clean_data, split_and_save_data
from src.features.build_features import prepare_features
from src.models.evaluate import evaluate_predictions
from src.utils.config import setup_logger, load_config

logger = setup_logger(__name__)


def build_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Instantiates candidate classification models based on configuration.
    """
    models_cfg = config.get("models", {})

    lr_params = models_cfg.get("logistic_regression", {})
    dt_params = models_cfg.get("decision_tree", {})
    rf_params = models_cfg.get("random_forest", {})

    models = {
        "logistic_regression": LogisticRegression(**lr_params),
        "decision_tree": DecisionTreeClassifier(**dt_params),
        "random_forest": RandomForestClassifier(**rf_params),
    }
    return models


def plot_feature_importances(
    model: Any,
    feature_names: list,
    model_name: str,
    output_path: str,
) -> None:
    """
    Plots and exports feature importances for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        palette="viridis",
    )
    plt.title(f"Feature Importances - {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info("Saved feature importances plot to %s", output_path)


def train_and_evaluate_all(
    config_path: str = "configs/config.yaml",
) -> Tuple[str, Any, Dict[str, Dict[str, Any]]]:
    """
    End-to-end training and evaluation pipeline for all candidate models.
    """
    config = load_config(config_path)
    raw_path = config["dataset"]["raw_path"]
    model_dir = Path(config.get("artifacts", {}).get("model_dir", "models/"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion & Validation
    df = load_raw_data(raw_path)
    cleaned_df = validate_and_clean_data(df, config)

    # 2. Split
    train_df, test_df = split_and_save_data(cleaned_df, config)

    # 3. Features & Preprocessing
    X_train, y_train, X_test, y_test, preprocessor = prepare_features(
        train_df, test_df, config, save_pipeline=True
    )
    feature_names = config["dataset"]["features"]["numerical"]

    # 4. Model Training & Evaluation
    models = build_models(config)
    results = {}
    best_model_name = ""
    best_f1 = -1.0
    best_model_obj = None

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)

        # Predictions on holdout test set
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, model_name=name)
        results[name] = metrics

        # Save individual model artifact
        model_save_path = model_dir / f"{name}.joblib"
        joblib.dump(model, model_save_path)

        # Plot feature importances if tree model
        plot_feature_importances(
            model=model,
            feature_names=feature_names,
            model_name=name,
            output_path=f"reports/figures/{name}_feature_importance.png",
        )

        # Track best model based on F1-score macro
        if metrics["f1_score_macro"] > best_f1:
            best_f1 = metrics["f1_score_macro"]
            best_model_name = name
            best_model_obj = model

    # Save best model
    best_path = model_dir / "best_model.joblib"
    joblib.dump(best_model_obj, best_path)
    logger.info("Selected '%s' as best model (F1=%.4f). Saved to %s", best_model_name, best_f1, best_path)

    # Save summary report
    summary_path = Path("reports/model_comparison.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = [
        {
            "Model": m["model"],
            "Accuracy": round(m["accuracy"], 4),
            "Precision (Macro)": round(m["precision_macro"], 4),
            "Recall (Macro)": round(m["recall_macro"], 4),
            "F1-Score (Macro)": round(m["f1_score_macro"], 4),
        }
        for m in results.values()
    ]
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    logger.info("Exported model comparison summary to %s", summary_path)

    return best_model_name, best_model_obj, results


if __name__ == "__main__":
    train_and_evaluate_all()
