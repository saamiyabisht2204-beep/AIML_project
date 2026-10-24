"""
Data loading, cleaning, validation, and splitting module.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import setup_logger, load_config

logger = setup_logger(__name__)


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw CSV data from the specified path.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    logger.info("Loading raw dataset from %s", file_path)
    df = pd.read_csv(path)
    logger.info("Loaded dataset with %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def validate_and_clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Validates required columns, removes invalid values and duplicates.
    """
    dataset_cfg = config.get("dataset", {})
    target_col = dataset_cfg.get("target_column", "Stress_Level")
    numerical_features = dataset_cfg.get("features", {}).get("numerical", [])
    valid_classes = set(dataset_cfg.get("target_classes", ["Low", "Moderate", "High"]))

    required_cols = numerical_features + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    cleaned_df = df.copy()

    # Drop duplicate records
    init_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    dropped_dupes = init_count - len(cleaned_df)
    if dropped_dupes > 0:
        logger.info("Removed %d duplicate rows", dropped_dupes)

    # Check and drop missing values
    missing_vals = cleaned_df[required_cols].isnull().sum().sum()
    if missing_vals > 0:
        logger.warning("Found %d missing values. Dropping incomplete rows.", missing_vals)
        cleaned_df = cleaned_df.dropna(subset=required_cols)

    # Filter invalid target categories
    valid_target_mask = cleaned_df[target_col].isin(valid_classes)
    invalid_targets = (~valid_target_mask).sum()
    if invalid_targets > 0:
        logger.warning("Dropping %d rows with unknown target values.", invalid_targets)
        cleaned_df = cleaned_df[valid_target_mask]

    logger.info(
        "Validation complete. Cleaned dataset has %d records. Target distribution:\n%s",
        len(cleaned_df),
        cleaned_df[target_col].value_counts().to_dict(),
    )
    return cleaned_df


def split_and_save_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train/test split and saves datasets to disk.
    """
    split_cfg = config.get("split", {})
    dataset_cfg = config.get("dataset", {})

    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)
    stratify_flag = split_cfg.get("stratify", True)
    target_col = dataset_cfg.get("target_column", "Stress_Level")

    stratify_target = df[target_col] if stratify_flag else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info("Data split: Train set=%d rows, Test set=%d rows", len(train_df), len(test_df))

    train_path = Path(dataset_cfg.get("processed_train_path", "data/processed/train.csv"))
    test_path = Path(dataset_cfg.get("processed_test_path", "data/processed/test.csv"))

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved train data to %s and test data to %s", train_path, test_path)
    return train_df, test_df


def run_data_pipeline(config_path: str = "configs/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes full ingestion, validation, and splitting pipeline.
    """
    config = load_config(config_path)
    raw_path = config["dataset"]["raw_path"]
    df = load_raw_data(raw_path)
    cleaned_df = validate_and_clean_data(df, config)
    return split_and_save_data(cleaned_df, config)


if __name__ == "__main__":
    run_data_pipeline()
