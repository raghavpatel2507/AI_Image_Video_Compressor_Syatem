"""
Machine Learning model predictor module.
Trains and serves RandomForest models for predicting optimal compression parameters.
Includes fallback content-adaptive heuristic logic.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from .image_features import FEATURE_COLUMNS as IMAGE_FEATURES
from .video_features import FEATURE_COLUMNS as VIDEO_FEATURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
IMAGE_MODEL_PATH = MODELS_DIR / "image_quality_model.joblib"
VIDEO_MODEL_PATH = MODELS_DIR / "video_crf_model.joblib"


def train_model(df, feature_columns: list, target_column: str, model_path: Path, n_estimators: int = 150):
    """Trains a RandomForestRegressor model and persists it to disk."""
    X = df[feature_columns].values
    y = df[target_column].values

    if len(df) < 5:
        raise ValueError(f"Insufficient training samples: got {len(df)}, need at least 5.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test) if len(X_test) > 0 else []
    mae = mean_absolute_error(y_test, preds) if len(preds) > 0 else float("nan")

    joblib.dump({"model": model, "features": feature_columns}, model_path)
    return model, mae


def train_image_model(df):
    """Trains the image quality predictor model."""
    return train_model(df, IMAGE_FEATURES, "optimal_quality", IMAGE_MODEL_PATH)


def train_video_model(df):
    """Trains the video CRF predictor model."""
    return train_model(df, VIDEO_FEATURES, "optimal_crf", VIDEO_MODEL_PATH)


def load_model_bundle(model_path: Path):
    """Loads model and feature schema bundle from file."""
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["features"]


def predict_image_quality(features: dict) -> tuple[int, str]:
    """Predicts optimal WebP image quality (45-95 range)."""
    loaded = load_model_bundle(IMAGE_MODEL_PATH)
    if loaded is not None:
        model, cols = loaded
        x = np.array([[features[c] for c in cols]])
        q = int(round(float(model.predict(x)[0])))
        return max(45, min(95, q)), "ml_model"

    # Fallback heuristic rule
    complexity = (
        0.4 * min(features["entropy"] / 8.0, 1.0)
        + 0.4 * min(features["edge_density"] * 20, 1.0)
        + 0.2 * min(features["laplacian_variance"] / 2000.0, 1.0)
    )
    quality = int(round(45 + complexity * 45))
    return max(45, min(95, quality)), "heuristic"


def predict_video_crf(features: dict) -> tuple[int, str]:
    """Predicts optimal H.264 video CRF (18-34 range)."""
    loaded = load_model_bundle(VIDEO_MODEL_PATH)
    if loaded is not None:
        model, cols = loaded
        x = np.array([[features[c] for c in cols]])
        crf = int(round(float(model.predict(x)[0])))
        return max(18, min(34, crf)), "ml_model"

    # Fallback heuristic rule
    complexity = (
        0.5 * min(features["avg_motion_magnitude"] / 5.0, 1.0)
        + 0.3 * min(features["avg_edge_density"] * 20, 1.0)
        + 0.2 * min(features["avg_entropy"] / 8.0, 1.0)
    )
    crf = int(round(32 - complexity * 12))
    return max(18, min(34, crf)), "heuristic"
