"""
Dataset generator for machine learning model training.
Sweeps compression parameters on sample media files to find optimal parameters
that satisfy target visual quality (SSIM).
"""

import argparse
import tempfile
from pathlib import Path
import pandas as pd

from .image_features import extract_image_features
from .video_features import extract_video_features
from .image_compressor import compress_image
from .video_compressor import compress_video
from .quality_metrics import evaluate_image_quality, evaluate_video_quality_ssim_psnr

QUALITY_LEVELS = [30, 40, 50, 60, 70, 80, 90]
CRF_LEVELS = [18, 22, 26, 30, 34, 38, 42]


def build_image_sample_row(image_path: str, target_ssim: float = 0.95) -> dict:
    features = extract_image_features(image_path)
    optimal_q = QUALITY_LEVELS[-1]

    with tempfile.TemporaryDirectory() as tmp:
        for q in QUALITY_LEVELS:
            out_path = str(Path(tmp) / f"temp_{q}.webp")
            compress_image(image_path, out_path, quality=q, fmt="webp")
            metrics = evaluate_image_quality(image_path, out_path)
            if metrics["ssim"] >= target_ssim:
                optimal_q = q
                break

    return {**features, "optimal_quality": optimal_q, "source_file": Path(image_path).name}


def build_video_sample_row(video_path: str, target_ssim: float = 0.95) -> dict:
    features = extract_video_features(video_path)
    optimal_crf = CRF_LEVELS[0]

    with tempfile.TemporaryDirectory() as tmp:
        for crf in reversed(CRF_LEVELS):
            out_path = str(Path(tmp) / f"temp_{crf}.mp4")
            compress_video(video_path, out_path, crf=crf, codec="libx264", preset="fast")
            metrics = evaluate_video_quality_ssim_psnr(video_path, out_path)
            if metrics["ssim"] is not None and metrics["ssim"] >= target_ssim:
                optimal_crf = crf
                break

    return {**features, "optimal_crf": optimal_crf, "source_file": Path(video_path).name}


def generate_image_dataset(image_dir: str, out_csv: str, target_ssim: float = 0.95):
    rows = []
    for p in sorted(Path(image_dir).glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            print(f"[Image Data Sweep] Processing {p.name} ...")
            rows.append(build_image_sample_row(str(p), target_ssim))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Dataset generated: {len(df)} rows -> {out_csv}")
    return df


def generate_video_dataset(video_dir: str, out_csv: str, target_ssim: float = 0.95):
    rows = []
    for p in sorted(Path(video_dir).glob("*")):
        if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
            print(f"[Video Data Sweep] Processing {p.name} ...")
            rows.append(build_video_sample_row(str(p), target_ssim))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Dataset generated: {len(df)} rows -> {out_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labeled dataset for compression ML model training.")
    parser.add_argument("--images", help="Directory path of image dataset")
    parser.add_argument("--videos", help="Directory path of video dataset")
    parser.add_argument("--out-dir", default="data", help="Output directory for generated CSVs")
    parser.add_argument("--target-ssim", type=float, default=0.95, help="Target SSIM score threshold")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.images:
        generate_image_dataset(args.images, f"{args.out_dir}/image_training_data.csv", args.target_ssim)
    if args.videos:
        generate_video_dataset(args.videos, f"{args.out_dir}/video_training_data.csv", args.target_ssim)
