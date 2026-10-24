"""
Model inference pipeline for individual and batch student stress predictions.
"""

from pathlib import Path
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd

from src.features.build_features import FeaturePipeline, INT_TO_LABEL
from src.utils.config import setup_logger

logger = setup_logger(__name__)


class StressPredictor:
    """
    Handles preprocessing and model inference for stress level prediction.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        preprocessor_path: str = "models/preprocessor.joblib",
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. Please train the models first."
            )
        if not self.preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Preprocessor artifact not found at '{preprocessor_path}'. Please train the models first."
            )

        logger.info("Loading model from %s and preprocessor from %s", model_path, preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.preprocessor: FeaturePipeline = FeaturePipeline.load(str(self.preprocessor_path))

    def predict_single(self, input_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicts stress category and confidence for an individual student.
        """
        required_features = self.preprocessor.numerical_features
        missing = [f for f in required_features if f not in input_features]
        if missing:
            raise ValueError(f"Missing required input features: {missing}")

        df = pd.DataFrame([input_features])[required_features]
        X_scaled = self.preprocessor.transform(df)

        pred_idx = int(self.model.predict(X_scaled)[0])
        pred_label = INT_TO_LABEL[pred_idx]

        probabilities = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0]
            for idx, prob in enumerate(probs):
                probabilities[INT_TO_LABEL[idx]] = round(float(prob), 4)
            confidence = probabilities[pred_label]
        else:
            confidence = 1.0

        return {
            "predicted_stress_level": pred_label,
            "confidence": confidence,
            "class_probabilities": probabilities,
            "input_features": input_features,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts stress category for a batch of student records.
        """
        required_features = self.preprocessor.numerical_features
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            raise ValueError(f"Batch DataFrame missing required columns: {missing}")

        X_scaled = self.preprocessor.transform(df[required_features])
        pred_indices = self.model.predict(X_scaled)
        pred_labels = [INT_TO_LABEL[idx] for idx in pred_indices]

        output_df = df.copy()
        output_df["Predicted_Stress_Level"] = pred_labels

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            for idx, label in INT_TO_LABEL.items():
                output_df[f"Prob_{label}"] = np.round(probs[:, idx], 4)

        return output_df
