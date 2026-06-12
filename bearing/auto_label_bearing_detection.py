import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-label bearing untuk object detection format YOLO."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset"),
        help="Folder dataset sumber yang berisi train/valid/test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_detection"),
        help="Folder output YOLO detection.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Split dataset yang diproses.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.18,
        help="Pengali ukuran bounding box dari radius bearing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa output yang sudah ada.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Simpan gambar preview dengan bounding box.",
    )
    parser.add_argument(
        "--class-csv",
        default="_classes.csv",
        help="Nama file CSV class classification di tiap split.",
    )
    return parser.parse_args()


def list_images(split_dir):
    return sorted(
        path
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_class_map(csv_path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not reader.fieldnames or "filename" not in reader.fieldnames:
        raise ValueError(f"Format CSV tidak valid: {csv_path}")

    class_names = [field for field in reader.fieldnames if field != "filename"]
    class_by_filename = {}

    for row in rows:
        active_classes = [
            class_name
            for class_name in class_names
            if row.get(class_name, "0").strip() == "1"
        ]
        if len(active_classes) != 1:
            raise ValueError(
                f"{csv_path}: {row['filename']} punya {len(active_classes)} class aktif."
            )
        class_by_filename[row["filename"]] = active_classes[0]

    return class_names, class_by_filename


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def circle_to_box(circle, width, height, padding):
    x, y, radius = circle
    half_size = radius * padding
    x1 = clamp(x - half_size, 0, width - 1)
    y1 = clamp(y - half_size, 0, height - 1)
    x2 = clamp(x + half_size, 0, width - 1)
    y2 = clamp(y + half_size, 0, height - 1)
    return x1, y1, x2, y2


def box_to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = x1 + (box_width / 2)
    center_y = y1 + (box_height / 2)

    return (
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height,
    )


def detect_with_hough(gray):
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.8)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(width, height) // 3,
        param1=80,
        param2=22,
        minRadius=max(8, min(width, height) // 18),
        maxRadius=max(18, min(width, height) // 3),
    )

    if circles is None:
        return None

    candidates = np.round(circles[0]).astype(int)
    image_center = np.array([width / 2, height / 2])

    def score(circle):
        x, y, radius = circle
        distance = np.linalg.norm(np.array([x, y]) - image_center)
        return radius - (distance * 0.08)

    return max(candidates, key=score)


def detect_with_contours(gray):
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 130)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = (width * height) * 0.003
    max_area = (width * height) * 0.45
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h else 0
        if 0.55 <= aspect_ratio <= 1.8:
            candidates.append((x, y, w, h, area))

    if not candidates:
        return None

    x, y, w, h, _ = max(candidates, key=lambda item: item[4])
    radius = max(w, h) / 2
    return x + (w / 2), y + (h / 2), radius


def detect_bearing(image_path, padding):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circle = detect_with_hough(gray)
    if circle is None:
        circle = detect_with_contours(gray)

    if circle is None:
        return image, None

    box = circle_to_box(circle, width, height, padding)
    return image, box


def write_label(label_path, box, width, height, class_id):
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if box is None:
        label_path.write_text("", encoding="utf-8")
        return

    values = box_to_yolo(box, width, height)
    line = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, *values)
    label_path.write_text(line, encoding="utf-8")


def draw_preview(image, box, class_name):
    preview = image.copy()
    if box is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            class_name,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return preview


def write_data_yaml(output_dir, class_names):
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    lines.append("")
    content = "\n".join(lines)
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def process_split(
    source_dir,
    output_dir,
    split,
    padding,
    overwrite,
    preview,
    class_csv,
    expected_class_names,
):
    split_dir = source_dir / split
    if not split_dir.exists():
        print(f"[skip] Split tidak ditemukan: {split_dir}")
        return expected_class_names, 0, 0

    csv_path = split_dir / class_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"File class tidak ditemukan: {csv_path}")

    class_names, class_by_filename = read_class_map(csv_path)
    if expected_class_names is not None and class_names != expected_class_names:
        raise ValueError(
            f"Urutan class di {csv_path} berbeda. "
            f"Expected {expected_class_names}, got {class_names}"
        )

    class_to_id = {class_name: index for index, class_name in enumerate(class_names)}

    image_paths = list_images(split_dir)
    detected_count = 0
    empty_count = 0

    for image_path in image_paths:
        output_image = output_dir / "images" / split / image_path.name
        output_label = output_dir / "labels" / split / f"{image_path.stem}.txt"

        if output_image.exists() and output_label.exists() and not overwrite:
            continue

        if image_path.name not in class_by_filename:
            raise ValueError(f"Label class tidak ditemukan untuk: {image_path}")

        class_name = class_by_filename[image_path.name]
        class_id = class_to_id[class_name]

        image, box = detect_bearing(image_path, padding)
        height, width = image.shape[:2]

        output_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_image)
        write_label(output_label, box, width, height, class_id)

        if preview:
            preview_image = draw_preview(image, box, class_name)
            preview_path = output_dir / "previews" / split / image_path.name
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), preview_image)

        if box is None:
            empty_count += 1
        else:
            detected_count += 1

    return class_names, detected_count, empty_count


def main():
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Folder source tidak ditemukan: {args.source}")

    totals = {}
    class_names = None
    for split in args.splits:
        class_names, detected_count, empty_count = process_split(
            args.source,
            args.output,
            split,
            args.padding,
            args.overwrite,
            args.preview,
            args.class_csv,
            class_names,
        )
        totals[split] = detected_count, empty_count

    if class_names is None:
        raise ValueError("Tidak ada split yang berhasil diproses.")

    write_data_yaml(args.output, class_names)

    print(f"Output: {args.output}")
    print("Class:")
    for index, class_name in enumerate(class_names):
        print(f"  {index} = {class_name}")
    for split, (detected_count, empty_count) in totals.items():
        print(f"{split}: {detected_count} label, {empty_count} kosong")


if __name__ == "__main__":
    main()
