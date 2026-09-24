"""
Quality metrics evaluation module.
Computes SSIM, PSNR (via scikit-image) and optional VMAF (via FFmpeg libvmaf filter).
"""

import subprocess
import shutil
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def check_vmaf_support() -> bool:
    """Checks if installed FFmpeg binary supports the libvmaf filter."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=10)
        return "libvmaf" in out.stdout
    except Exception:
        return False


VMAF_SUPPORTED = check_vmaf_support()


def evaluate_image_quality(original_path: str, compressed_path: str) -> dict:
    """Computes SSIM and PSNR between original and compressed image."""
    orig = cv2.imread(original_path, cv2.IMREAD_COLOR)
    comp = cv2.imread(compressed_path, cv2.IMREAD_COLOR)
    if orig is None or comp is None:
        raise ValueError("Could not read original or compressed image file.")

    if comp.shape != orig.shape:
        comp = cv2.resize(comp, (orig.shape[1], orig.shape[0]))

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    comp_gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)

    ssim_score = float(ssim(orig_gray, comp_gray))
    psnr_score = float(psnr(orig, comp, data_range=255))

    return {"ssim": round(ssim_score, 4), "psnr": round(psnr_score, 2)}


def sample_video_frames(path: str, n_frames: int = 10):
    """Extracts evenly spaced frame samples from a video file."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), min(n_frames, max(total, 1)), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def evaluate_video_quality_ssim_psnr(original_path: str, compressed_path: str, n_samples: int = 10) -> dict:
    """Calculates frame-averaged SSIM and PSNR for videos."""
    orig_frames = sample_video_frames(original_path, n_samples)
    comp_frames = sample_video_frames(compressed_path, n_samples)
    n = min(len(orig_frames), len(comp_frames))
    if n == 0:
        return {"ssim": None, "psnr": None}

    ssim_scores, psnr_scores = [], []
    for i in range(n):
        o, c = orig_frames[i], comp_frames[i]
        if c.shape != o.shape:
            c = cv2.resize(c, (o.shape[1], o.shape[0]))
        og, cg = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        ssim_scores.append(ssim(og, cg))
        psnr_scores.append(psnr(o, c, data_range=255))

    return {
        "ssim": round(float(np.mean(ssim_scores)), 4),
        "psnr": round(float(np.mean(psnr_scores)), 2),
    }


def evaluate_video_quality_vmaf(original_path: str, compressed_path: str) -> float | None:
    """Calculates mean VMAF score via FFmpeg if libvmaf filter is available."""
    if not VMAF_SUPPORTED:
        return None
    cmd = [
        "ffmpeg", "-i", compressed_path, "-i", original_path,
        "-lavfi", "[0:v][1:v]libvmaf",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stderr.splitlines():
        if "VMAF score" in line:
            try:
                return float(line.strip().split("VMAF score:")[-1].strip())
            except ValueError:
                pass
    return None


def evaluate_video_quality(original_path: str, compressed_path: str) -> dict:
    """Evaluates video quality metrics (SSIM, PSNR, VMAF)."""
    metrics = evaluate_video_quality_ssim_psnr(original_path, compressed_path)
    metrics["vmaf"] = evaluate_video_quality_vmaf(original_path, compressed_path)
    return metrics
