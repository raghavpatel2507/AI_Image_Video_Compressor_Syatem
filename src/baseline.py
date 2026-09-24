from .image_compressor import compress_image
from .video_compressor import compress_video

FIXED_IMAGE_QUALITY = 75   
FIXED_VIDEO_CRF = 23      


def baseline_compress_image(input_path: str, output_path: str) -> dict:
    return compress_image(input_path, output_path, quality=FIXED_IMAGE_QUALITY, fmt="webp")


def baseline_compress_video(input_path: str, output_path: str) -> dict:
    return compress_video(input_path, output_path, crf=FIXED_VIDEO_CRF, codec="libx264", preset="medium")
