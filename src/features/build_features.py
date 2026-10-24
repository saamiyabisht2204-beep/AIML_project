"""
Feature engineering, scaling, and transformation pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils.config import setup_logger, load_config

logger = setup_logger(__name__)

# Canonical class mapping
LABEL_TO_INT = {"Low": 0, "Moderate": 1, "High": 2}
INT_TO_LABEL = {0: "Low", 1: "Moderate", 2: "High"}


class FeaturePipeline:
    """
    Encapsulates feature scaling and target transformations.
    """

    def __init__(self, method: str = "standard", numerical_features: list = None):
        self.method = method
        self.numerical_features = numerical_features or [
            "Study_Hours_Per_Day",
            "Extracurricular_Hours_Per_Day",
            "Sleep_Hours_Per_Day",
            "Social_Hours_Per_Day",
            "Physical_Activity_Hours_Per_Day",
            "GPA",
        ]
        if self.method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """
        Fits the scaler on the numerical training features.
        """
        logger.info("Fitting %s scaler on %d features", self.method, len(self.numerical_features))
        X_num = df[self.numerical_features].values
        self.scaler.fit(X_num)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame into scaled feature array.
        """
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform.")
        X_num = df[self.numerical_features].values
        return self.scaler.transform(X_num)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits and transforms numerical features in one step.
        """
        return self.fit(df).transform(df)

    def save(self, filepath: str = "models/preprocessor.joblib") -> None:
        """
        Serializes pipeline to disk.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved fitted preprocessor to %s", filepath)

    @classmethod
    def load(cls, filepath: str = "models/preprocessor.joblib") -> "FeaturePipeline":
        """
        Loads fitted pipeline from disk.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        return joblib.load(path)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    save_pipeline: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeaturePipeline]:
    """
    Extracts features and target labels, applies scaling, and returns numpy arrays.
    """
    dataset_cfg = config.get("dataset", {})
    scaling_cfg = config.get("scaling", {})

    target_col = dataset_cfg.get("target_column", "Stress_Level")
    numerical_features = dataset_cfg.get("features", {}).get("numerical", [])
    method = scaling_cfg.get("method", "standard")

    pipeline = FeaturePipeline(method=method, numerical_features=numerical_features)

    # Transform features
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)

    # Encode targets
    y_train = train_df[target_col].map(LABEL_TO_INT).values
    y_test = test_df[target_col].map(LABEL_TO_INT).values

    if save_pipeline:
        model_dir = config.get("artifacts", {}).get("model_dir", "models/")
        pipeline.save(str(Path(model_dir) / "preprocessor.joblib"))

    logger.info("Features prepared: X_train shape=%s, X_test shape=%s", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test, pipeline
