"""
video_compressor.py
Compresses a video via FFmpeg using a given CRF (Constant Rate Factor) and
codec. Lower CRF = higher quality/larger file, higher CRF = more compression.

Codec CRF ranges (approx):
    libx264 : 0 (lossless) - 51 (worst)   -> visually good range ~18-30
    libx265 : 0 - 51                       -> visually good range ~20-32 (roughly
                                               2-3 CRF points "better" than x264 at
                                               same visual quality, i.e. smaller files)
"""

import subprocess
import shutil
from pathlib import Path


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def compress_video(
    input_path: str,
    output_path: str,
    crf: int,
    codec: str = "libx264",
    preset: str = "medium",
) -> dict:
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH. Install it (see README).")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    crf = int(max(0, min(51, crf)))

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg process timed out while processing '{input_path}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to execute FFmpeg command for '{input_path}': {e}")

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg compression failed (crf={crf}, codec={codec}):\n{result.stderr[-2000:]}")

    return {
        "output_path": str(out),
        "codec": codec,
        "crf": crf,
        "preset": preset,
        "output_size_bytes": out.stat().st_size,
    }
