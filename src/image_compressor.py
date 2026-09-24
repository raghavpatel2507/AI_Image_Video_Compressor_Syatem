from pathlib import Path
from PIL import Image

try:
    import pillow_avif 
    AVIF_AVAILABLE = True
except ImportError:
    AVIF_AVAILABLE = False


def compress_image(input_path: str, output_path: str, quality: int, fmt: str = "webp") -> dict:
    """
    quality: 1-100 (higher = better quality, larger file)
    fmt: 'jpeg' | 'webp' | 'avif'
    """
    fmt = fmt.lower()
    if fmt == "avif" and not AVIF_AVAILABLE:
        fmt = "webp"  

    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open or decode image '{input_path}': {e}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    quality = int(max(1, min(100, quality)))

    if fmt == "jpeg":
        img.save(out, "JPEG", quality=quality, optimize=True)
    elif fmt == "webp":
        img.save(out, "WEBP", quality=quality, method=6)
    elif fmt == "avif":
        img.save(out, "AVIF", quality=quality)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return {
        "output_path": str(out),
        "format": fmt,
        "quality": quality,
        "output_size_bytes": out.stat().st_size,
    }
