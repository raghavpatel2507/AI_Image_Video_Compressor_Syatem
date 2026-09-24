"""
Video feature extraction module.
Samples frames across a video to compute optical flow motion magnitude,
edge density, and grayscale entropy.
"""

from pathlib import Path
import cv2
import numpy as np


def compute_entropy(gray: np.ndarray) -> float:
    """Computes Shannon entropy for a grayscale frame histogram."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def extract_video_features(video_path: str, max_samples: int = 12) -> dict:
    """Extracts motion and visual complexity features from a video file."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    sample_idxs = np.linspace(0, max(frame_count - 1, 0), min(max_samples, max(frame_count, 1)), dtype=int)

    prev_gray = None
    motion_scores, edge_scores, entropy_scores = [], [], []

    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)
        edge_scores.append(float(np.count_nonzero(edges)) / (gray.shape[0] * gray.shape[1]))
        entropy_scores.append(compute_entropy(gray))

        if prev_gray is not None and prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_scores.append(float(np.mean(mag)))
        prev_gray = gray

    cap.release()

    return {
        "width": w,
        "height": h,
        "resolution_mp": round((h * w) / 1e6, 3),
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 2),
        "avg_motion_magnitude": round(float(np.mean(motion_scores)) if motion_scores else 0.0, 4),
        "avg_edge_density": round(float(np.mean(edge_scores)) if edge_scores else 0.0, 5),
        "avg_entropy": round(float(np.mean(entropy_scores)) if entropy_scores else 0.0, 4),
        "file_size_bytes": path.stat().st_size,
    }


FEATURE_COLUMNS = [
    "resolution_mp",
    "fps",
    "duration_sec",
    "avg_motion_magnitude",
    "avg_edge_density",
    "avg_entropy",
]

get_video_features = extract_video_features
