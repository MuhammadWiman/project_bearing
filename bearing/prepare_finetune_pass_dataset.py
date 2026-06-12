import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from auto_label_bearing_detection import (
    IMAGE_SUFFIXES,
    circle_to_box,
    detect_bearing,
    draw_preview,
    write_data_yaml,
    write_label,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["Reject", "Pass"]
PASS_CLASS_ID = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Siapkan dataset fine-tuning YOLO dari dataset lama + train_tambahan_pass."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("dataset_detection_2class"),
        help="Dataset YOLO 2 class lama.",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        default=Path("train_tambahan_pass"),
        help="Folder gambar tambahan class Pass.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_detection_2class_finetune_pass_aug"),
        help="Folder output dataset fine-tuning.",
    )
    parser.add_argument("--padding", type=float, default=1.18)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-base-copy",
        action="store_true",
        help="Lewati salin dataset base, berguna untuk melanjutkan proses yang timeout.",
    )
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def copy_base_dataset(base_dir, output_dir, overwrite):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output sudah ada: {output_dir}. Pakai --overwrite untuk membuat ulang."
            )
        shutil.rmtree(output_dir)

    for folder in ["images", "labels"]:
        source = base_dir / folder
        target = output_dir / folder
        if source.exists():
            shutil.copytree(source, target)


def list_extra_images(extra_dir):
    return sorted(
        path
        for path in extra_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def save_image_and_label(image, box, output_image, output_label):
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_image), image)
    height, width = image.shape[:2]
    write_label(output_label, box, width, height, PASS_CLASS_ID)


def add_preview(image, box, preview_path):
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), draw_preview(image, box, "Pass"))


def detect_extra_bearing(image_path, padding):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.8)

    # Gambar tambahan beresolusi tinggi dan beberapa bearing besar berada di
    # tepi frame. Ambil kandidat Hough dengan confidence tertinggi agar tidak
    # memilih lingkaran palsu besar dari tekstur alas hijau.
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(width, height) // 3,
        param1=100,
        param2=28,
        minRadius=max(60, min(width, height) // 12),
        maxRadius=max(120, min(width, height) // 4),
    )

    if circles is not None:
        x, y, radius = np.round(circles[0][0]).astype(int)
        return image, circle_to_box((x, y, radius), width, height, padding)

    return detect_bearing(image_path, padding)


def flip_horizontal(image, box):
    flipped = cv2.flip(image, 1)
    if box is None:
        return flipped, None

    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    return flipped, (width - x2, y1, width - x1, y2)


def adjust_brightness(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_extra_images(extra_dir, output_dir, padding, preview):
    detected_count = 0
    empty_count = 0

    for image_path in list_extra_images(extra_dir):
        image, box = detect_extra_bearing(image_path, padding)
        if box is None:
            empty_count += 1
        else:
            detected_count += 1

        variants = [
            ("orig", image, box),
            ("flip", *flip_horizontal(image, box)),
            ("bright", adjust_brightness(image, 1.12, 12), box),
            ("dim", adjust_brightness(image, 0.88, -8), box),
        ]

        for suffix, variant_image, variant_box in variants:
            name = f"{image_path.stem}_{suffix}{image_path.suffix.lower()}"
            label_name = f"{image_path.stem}_{suffix}.txt"
            output_image = output_dir / "images" / "train" / name
            output_label = output_dir / "labels" / "train" / label_name
            save_image_and_label(variant_image, variant_box, output_image, output_label)

            if preview:
                add_preview(
                    variant_image,
                    variant_box,
                    output_dir / "previews" / "train_tambahan_pass" / name,
                )

    return detected_count, empty_count


def write_dataset_yaml(output_dir):
    write_data_yaml(output_dir, CLASS_NAMES)


def main():
    args = parse_args()
    base_dir = resolve(args.base)
    extra_dir = resolve(args.extra)
    output_dir = resolve(args.output)

    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset base tidak ditemukan: {base_dir}")
    if not extra_dir.exists():
        raise FileNotFoundError(f"Folder tambahan tidak ditemukan: {extra_dir}")

    if not args.skip_base_copy:
        copy_base_dataset(base_dir, output_dir, args.overwrite)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    detected_count, empty_count = add_extra_images(
        extra_dir, output_dir, args.padding, args.preview
    )
    write_dataset_yaml(output_dir)

    extra_count = len(list_extra_images(extra_dir))
    print(f"Output: {output_dir}")
    print(f"Gambar tambahan: {extra_count}")
    print(f"Auto-label terdeteksi: {detected_count}")
    print(f"Auto-label kosong: {empty_count}")
    print(f"Augmentasi tambahan train: {extra_count * 4}")
    print(f"Data YAML: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
