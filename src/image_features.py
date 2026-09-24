"""
Image feature extraction module.
Extracts visual complexity metrics (entropy, edge density, color variance, sharpness)
to drive adaptive compression.
"""

from pathlib import Path
import cv2
import numpy as np


def compute_entropy(gray: np.ndarray) -> float:
    """Computes Shannon entropy for a grayscale image histogram."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def extract_image_features(image_path: str) -> dict:
    """Extracts structural and complexity features from an image file."""
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    entropy_val = compute_entropy(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / (h * w)
    color_var = float(np.var(img.astype(np.float32)))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    res_mp = (h * w) / 1e6

    return {
        "width": w,
        "height": h,
        "resolution_mp": round(res_mp, 3),
        "entropy": round(entropy_val, 4),
        "edge_density": round(edge_density, 5),
        "color_variance": round(color_var, 3),
        "laplacian_variance": round(laplacian_var, 3),
        "file_size_bytes": path.stat().st_size,
    }


FEATURE_COLUMNS = [
    "resolution_mp",
    "entropy",
    "edge_density",
    "color_variance",
    "laplacian_variance",
]

get_image_features = extract_image_features
