"""
Adaptive compression pipeline module.
Extracts features, predicts compression parameters via ML/Heuristic engine,
runs compression, evaluates SSIM/PSNR/VMAF, and executes optimization loops.
"""

import time
from pathlib import Path

from .image_features import extract_image_features
from .video_features import extract_video_features
from .image_compressor import compress_image
from .video_compressor import compress_video
from .quality_metrics import evaluate_image_quality, evaluate_video_quality
from .ml_predictor import predict_image_quality, predict_video_crf

TARGET_SSIM = 0.95
MAX_ITERATIONS = 3


def compress_image_adaptive(input_path: str, output_path: str, target_ssim: float = TARGET_SSIM) -> dict:
    """Runs adaptive image compression with dynamic quality optimization."""
    t0 = time.time()
    features = extract_image_features(input_path)
    quality, source = predict_image_quality(features)

    history = []
    for iteration in range(1, MAX_ITERATIONS + 1):
        comp = compress_image(input_path, output_path, quality=quality, fmt="webp")
        qual = evaluate_image_quality(input_path, output_path)
        history.append({"iteration": iteration, "quality_param": quality, **qual})

        if qual["ssim"] >= target_ssim:
            if qual["ssim"] - target_ssim > 0.03 and quality > 60:
                quality = max(45, quality - 5)
                continue
            break
        else:
            quality = min(95, quality + 10)

    elapsed = round(time.time() - t0, 3)
    orig_size = Path(input_path).stat().st_size
    out_size = Path(output_path).stat().st_size

    return {
        "file": Path(input_path).name,
        "type": "image",
        "approach": "ai_adaptive",
        "param_source": source,
        "final_quality_param": history[-1]["quality_param"],
        "iterations": len(history),
        "original_size_bytes": orig_size,
        "compressed_size_bytes": out_size,
        "size_reduction_pct": round(100 * (1 - out_size / orig_size), 2),
        "ssim": history[-1]["ssim"],
        "psnr": history[-1]["psnr"],
        "processing_time_sec": elapsed,
        "history": history,
        "features": features,
    }


def compress_video_adaptive(input_path: str, output_path: str, target_ssim: float = TARGET_SSIM) -> dict:
    """Runs adaptive video compression with dynamic CRF optimization."""
    t0 = time.time()
    features = extract_video_features(input_path)
    crf, source = predict_video_crf(features)

    history = []
    for iteration in range(1, MAX_ITERATIONS + 1):
        comp = compress_video(input_path, output_path, crf=crf, codec="libx264", preset="medium")
        qual = evaluate_video_quality(input_path, output_path)
        history.append({"iteration": iteration, "crf": crf, **qual})

        if qual["ssim"] is not None and qual["ssim"] >= target_ssim:
            if qual["ssim"] - target_ssim > 0.03 and crf < 30:
                crf = min(34, crf + 3)
                continue
            break
        else:
            crf = max(18, crf - 4)

    elapsed = round(time.time() - t0, 3)
    orig_size = Path(input_path).stat().st_size
    out_size = Path(output_path).stat().st_size

    return {
        "file": Path(input_path).name,
        "type": "video",
        "approach": "ai_adaptive",
        "param_source": source,
        "final_crf": history[-1]["crf"],
        "iterations": len(history),
        "original_size_bytes": orig_size,
        "compressed_size_bytes": out_size,
        "size_reduction_pct": round(100 * (1 - out_size / orig_size), 2),
        "ssim": history[-1]["ssim"],
        "psnr": history[-1]["psnr"],
        "vmaf": history[-1].get("vmaf"),
        "processing_time_sec": elapsed,
        "history": history,
        "features": features,
    }


run_adaptive_image_compression = compress_image_adaptive
run_adaptive_video_compression = compress_video_adaptive
