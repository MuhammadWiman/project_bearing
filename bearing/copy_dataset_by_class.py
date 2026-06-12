import argparse
import csv
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Salin dataset Roboflow ke folder terpisah berdasarkan class."
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
        default=Path("dataset_by_class"),
        help="Folder output hasil salinan.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Nama class yang ingin disalin. Default: semua class di _classes.csv.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Split dataset yang diproses.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa file jika sudah ada di output.",
    )
    return parser.parse_args()


def read_class_rows(csv_path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not reader.fieldnames or "filename" not in reader.fieldnames:
        raise ValueError(f"Format CSV tidak valid: {csv_path}")

    classes = [field for field in reader.fieldnames if field != "filename"]
    return classes, rows


def copy_image(source_path, target_path, overwrite):
    if target_path.exists() and not overwrite:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return True


def copy_split(source_dir, output_dir, split, selected_classes, overwrite):
    split_dir = source_dir / split
    csv_path = split_dir / "_classes.csv"

    if not split_dir.exists():
        print(f"[skip] Split tidak ditemukan: {split_dir}")
        return {}

    if not csv_path.exists():
        print(f"[skip] File label tidak ditemukan: {csv_path}")
        return {}

    available_classes, rows = read_class_rows(csv_path)
    classes = selected_classes or available_classes
    unknown_classes = [label for label in classes if label not in available_classes]

    if unknown_classes:
        raise ValueError(
            f"Class tidak ada di {csv_path}: {', '.join(unknown_classes)}"
        )

    counts = {label: 0 for label in classes}
    skipped_missing = 0

    for row in rows:
        filename = row["filename"]
        source_path = split_dir / filename

        if not source_path.exists() or source_path.suffix.lower() not in IMAGE_SUFFIXES:
            skipped_missing += 1
            continue

        for label in classes:
            if row.get(label, "0").strip() != "1":
                continue

            target_path = output_dir / split / label / filename
            copied = copy_image(source_path, target_path, overwrite)
            if copied:
                counts[label] += 1

    if skipped_missing:
        print(f"[warn] {split}: {skipped_missing} gambar tidak ditemukan / bukan gambar.")

    return counts


def main():
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Folder source tidak ditemukan: {args.source}")

    total_counts = {}

    for split in args.splits:
        counts = copy_split(
            args.source,
            args.output,
            split,
            args.classes,
            args.overwrite,
        )
        total_counts[split] = counts

    print(f"Output: {args.output}")
    for split, counts in total_counts.items():
        if not counts:
            continue

        print(f"{split}:")
        for label, count in counts.items():
            print(f"  - {label}: {count} file")


if __name__ == "__main__":
    main()
