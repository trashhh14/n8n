#!/usr/bin/env python3
"""
Filter a directory of photos by similarity to a reference face image.

Requirements:
  pip install face_recognition pillow

Example:
  python scripts/face_filter.py \
    --reference ./reference.jpg \
    --input-dir ./photos \
    --output-dir ./matched \
    --tolerance 0.5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import face_recognition

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(path: Path) -> Iterable[Path]:
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield item


def load_reference_encoding(reference_path: Path) -> List[float]:
    image = face_recognition.load_image_file(reference_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise ValueError("No face found in reference image.")
    return encodings[0]


def image_matches_reference(
    image_path: Path,
    reference_encoding: List[float],
    tolerance: float,
) -> bool:
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return False
    matches = face_recognition.compare_faces(encodings, reference_encoding, tolerance=tolerance)
    return any(matches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter photos by face similarity.")
    parser.add_argument("--reference", required=True, type=Path, help="Path to reference face image.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with candidate photos.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for matched photos.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Face match tolerance (lower = stricter). Default: 0.6",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move matched photos instead of copying.",
    )
    args = parser.parse_args()

    if not args.reference.exists():
        raise FileNotFoundError(f"Reference image not found: {args.reference}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_encoding = load_reference_encoding(args.reference)

    for image_path in find_images(args.input_dir):
        if image_matches_reference(image_path, reference_encoding, args.tolerance):
            destination = args.output_dir / image_path.name
            if args.move:
                shutil.move(str(image_path), destination)
            else:
                shutil.copy2(str(image_path), destination)


if __name__ == "__main__":
    main()
