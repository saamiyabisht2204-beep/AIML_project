"""
Unit and integration tests for the ML pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import load_config
from src.data.make_dataset import validate_and_clean_data
from src.features.build_features import FeaturePipeline, LABEL_TO_INT
from src.models.evaluate import evaluate_predictions
from src.models.train import build_models


@pytest.fixture
def sample_config():
    return {
        "dataset": {
            "target_column": "Stress_Level",
            "target_classes": ["Low", "Moderate", "High"],
            "features": {
                "numerical": [
                    "Study_Hours_Per_Day",
                    "Extracurricular_Hours_Per_Day",
                    "Sleep_Hours_Per_Day",
                    "Social_Hours_Per_Day",
                    "Physical_Activity_Hours_Per_Day",
                    "GPA",
                ]
            },
        },
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
        "scaling": {"method": "standard"},
        "models": {
            "logistic_regression": {"max_iter": 100, "random_state": 42},
            "decision_tree": {"max_depth": 3, "random_state": 42},
            "random_forest": {"n_estimators": 10, "max_depth": 3, "random_state": 42},
        },
    }


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "Student_ID": range(1, n + 1),
            "Study_Hours_Per_Day": np.random.uniform(2, 10, n),
            "Extracurricular_Hours_Per_Day": np.random.uniform(0, 5, n),
            "Sleep_Hours_Per_Day": np.random.uniform(4, 10, n),
            "Social_Hours_Per_Day": np.random.uniform(1, 5, n),
            "Physical_Activity_Hours_Per_Day": np.random.uniform(0, 4, n),
            "GPA": np.random.uniform(2.0, 4.0, n),
            "Stress_Level": np.random.choice(["Low", "Moderate", "High"], n),
        }
    )
    return df


def test_load_config():
    config = load_config("configs/config.yaml")
    assert "dataset" in config
    assert "models" in config
    assert config["dataset"]["target_column"] == "Stress_Level"


def test_validate_and_clean_data(sample_data, sample_config):
    cleaned = validate_and_clean_data(sample_data, sample_config)
    assert len(cleaned) == len(sample_data)
    assert set(cleaned["Stress_Level"].unique()).issubset({"Low", "Moderate", "High"})


def test_feature_pipeline(sample_data):
    pipeline = FeaturePipeline(method="standard")
    X = pipeline.fit_transform(sample_data)
    assert X.shape == (len(sample_data), 6)
    assert np.allclose(X.mean(axis=0), np.zeros(6), atol=1e-5)


def test_model_training_and_eval(sample_data, sample_config):
    models = build_models(sample_config)
    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(sample_data)
    y = sample_data["Stress_Level"].map(LABEL_TO_INT).values

    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        metrics = evaluate_predictions(y, preds, model_name=f"test_{name}", save_artifacts=False)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_score_macro"] <= 1.0
