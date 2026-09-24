"""
Report generator module.
Runs comparative benchmarking between traditional fixed compression
and adaptive ML compression across media files, producing CSV reports and summaries.
"""

import time
from pathlib import Path
import pandas as pd

from .baseline import baseline_compress_image, baseline_compress_video
from .quality_metrics import evaluate_image_quality, evaluate_video_quality
from .pipeline import compress_image_adaptive, compress_video_adaptive

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def run_comparison(input_dir: str, output_dir: str) -> pd.DataFrame:
    """Executes comparative compression benchmark over an input directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for p in sorted(input_dir.glob("*")):
        ext = p.suffix.lower()

        if ext in IMAGE_EXTS:
            t0 = time.time()
            base_out = output_dir / f"{p.stem}_baseline.webp"
            base_comp = baseline_compress_image(str(p), str(base_out))
            base_qual = evaluate_image_quality(str(p), str(base_out))
            base_time = round(time.time() - t0, 3)

            ai_out = output_dir / f"{p.stem}_ai.webp"
            ai_result = compress_image_adaptive(str(p), str(ai_out))

            orig_size = p.stat().st_size
            rows.append({
                "file": p.name,
                "type": "image",
                "original_size_kb": round(orig_size / 1024, 1),
                "baseline_size_kb": round(base_comp["output_size_bytes"] / 1024, 1),
                "baseline_reduction_pct": round(100 * (1 - base_comp["output_size_bytes"] / orig_size), 2),
                "baseline_ssim": base_qual["ssim"],
                "baseline_time_sec": base_time,
                "ai_size_kb": round(ai_result["compressed_size_bytes"] / 1024, 1),
                "ai_reduction_pct": ai_result["size_reduction_pct"],
                "ai_ssim": ai_result["ssim"],
                "ai_time_sec": ai_result["processing_time_sec"],
                "ai_param_source": ai_result["param_source"],
                "ai_iterations": ai_result["iterations"],
            })

        elif ext in VIDEO_EXTS:
            t0 = time.time()
            base_out = output_dir / f"{p.stem}_baseline.mp4"
            base_comp = baseline_compress_video(str(p), str(base_out))
            base_qual = evaluate_video_quality(str(p), str(base_out))
            base_time = round(time.time() - t0, 3)

            ai_out = output_dir / f"{p.stem}_ai.mp4"
            ai_result = compress_video_adaptive(str(p), str(ai_out))

            orig_size = p.stat().st_size
            rows.append({
                "file": p.name,
                "type": "video",
                "original_size_kb": round(orig_size / 1024, 1),
                "baseline_size_kb": round(base_comp["output_size_bytes"] / 1024, 1),
                "baseline_reduction_pct": round(100 * (1 - base_comp["output_size_bytes"] / orig_size), 2),
                "baseline_ssim": base_qual["ssim"],
                "baseline_time_sec": base_time,
                "ai_size_kb": round(ai_result["compressed_size_bytes"] / 1024, 1),
                "ai_reduction_pct": ai_result["size_reduction_pct"],
                "ai_ssim": ai_result["ssim"],
                "ai_time_sec": ai_result["processing_time_sec"],
                "ai_param_source": ai_result["param_source"],
                "ai_iterations": ai_result["iterations"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "comparison_report.csv", index=False)
    return df


def print_summary(df: pd.DataFrame):
    """Prints a clean summary of comparison metrics."""
    if df.empty:
        print("No files processed.")
        return
    print("\n" + "=" * 45)
    print("      COMPRESSION COMPARISON SUMMARY")
    print("=" * 45)
    for _, r in df.iterrows():
        print(f"\nFile: {r['file']} ({r['type']})")
        print(f"  Baseline : {r['baseline_size_kb']} KB  (-{r['baseline_reduction_pct']}%)  SSIM={r['baseline_ssim']}  ({r['baseline_time_sec']}s)")
        print(f"  AI Model : {r['ai_size_kb']} KB  (-{r['ai_reduction_pct']}%)  SSIM={r['ai_ssim']}  ({r['ai_time_sec']}s)  [{r['ai_param_source']}, {r['ai_iterations']} iter]")
    print("\n" + "-" * 45)
    print(f"Average Baseline Size Reduction : {df['baseline_reduction_pct'].mean():.2f}%")
    print(f"Average AI Model Size Reduction : {df['ai_reduction_pct'].mean():.2f}%")
    print("-" * 45 + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        df = run_comparison(sys.argv[1], sys.argv[2])
        print_summary(df)
