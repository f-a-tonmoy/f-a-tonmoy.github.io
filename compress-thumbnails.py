"""
Compress article thumbnail images.

Resizes PNGs in assets/article-thumbnails/ to a max width and converts them
to JPG, which typically reduces total size by ~95%. Also rewrites articles.js
so the .png references become .jpg.

Usage:
    pip install pillow
    python compress-thumbnails.py
    # or, to also delete the original PNGs after compression:
    python compress-thumbnails.py --delete-originals
"""

import argparse
import re
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is not installed. Run: pip install pillow")

ROOT = Path(__file__).parent
THUMB_DIR = ROOT / "assets" / "article-thumbnails"
ARTICLES_JS = ROOT / "articles.js"

MAX_WIDTH = 800       # thumbnails display at ~400px wide; 800px = retina-ready
JPG_QUALITY = 82      # 78-85 is the sweet spot for photos


def compress_one(src: Path) -> Path:
    """Resize + convert a single PNG to JPG. Returns the new file path."""
    img = Image.open(src)
    # Flatten transparency onto white so the JPG looks right
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    else:
        img = img.convert("RGB")

    if img.width > MAX_WIDTH:
        new_height = round(img.height * MAX_WIDTH / img.width)
        img = img.resize((MAX_WIDTH, new_height), Image.LANCZOS)

    dst = src.with_suffix(".jpg")
    img.save(dst, "JPEG", quality=JPG_QUALITY, optimize=True, progressive=True)
    return dst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete the original PNG files after successful compression.",
    )
    args = parser.parse_args()

    pngs = sorted(THUMB_DIR.glob("*.png"))
    if not pngs:
        sys.exit(f"No PNGs found in {THUMB_DIR}")

    print(f"Found {len(pngs)} PNG thumbnails. Compressing to JPG...\n")
    total_before = 0
    total_after = 0

    for src in pngs:
        try:
            before = src.stat().st_size
            dst = compress_one(src)
            after = dst.stat().st_size
        except Exception as exc:
            print(f"  [FAIL] {src.name}: {exc}")
            continue
        total_before += before
        total_after += after
        ratio = (1 - after / before) * 100
        print(
            f"  {src.name[:55]:55s}  "
            f"{before/1024:7.0f} KB -> {after/1024:7.0f} KB  "
            f"(-{ratio:4.0f}%)"
        )

    saved = total_before - total_after
    print(
        f"\nTotal: {total_before/1024/1024:.1f} MB -> {total_after/1024/1024:.1f} MB"
        f"   (saved {saved/1024/1024:.1f} MB, "
        f"-{(saved/total_before)*100:.0f}%)"
    )

    # Rewrite articles.js so .png references become .jpg
    if ARTICLES_JS.exists():
        text = ARTICLES_JS.read_text(encoding="utf-8")
        new_text = re.sub(
            r'(article-thumbnails/[^"\']+)\.png',
            r'\1.jpg',
            text,
        )
        if text != new_text:
            ARTICLES_JS.write_text(new_text, encoding="utf-8")
            print(f"Updated {ARTICLES_JS.name} (.png -> .jpg).")

    if args.delete_originals:
        for src in pngs:
            try:
                src.unlink()
            except FileNotFoundError:
                pass
        print(f"Deleted {len(pngs)} original PNG files.")
    else:
        print(
            "\nOriginal PNGs were kept. "
            "Re-run with --delete-originals once you've verified the output."
        )


if __name__ == "__main__":
    main()
