import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO_CONFIG_DIR = PROJECT_ROOT

os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_YOLO_CONFIG_DIR))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 untuk dataset object detection bearing 2 class."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset_detection_2class/data.yaml"),
        help="Path data.yaml dataset YOLO.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Model awal YOLOv8, misalnya yolov8n.pt, yolov8s.pt, atau path .pt lokal.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Jumlah epoch training.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Ukuran input image.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device training. Contoh: cpu, 0, atau 0,1. Default auto.",
    )
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Folder output training.",
    )
    parser.add_argument(
        "--name",
        default="bearing_yolov8",
        help="Nama run training.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Jumlah dataloader workers. Di Windows, 0 biasanya paling stabil.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Lanjutkan training dari run terakhir jika tersedia.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Jalankan validasi setelah training selesai.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Jalankan evaluasi pada split test setelah training selesai.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main():
    args = parse_args()
    data_path = resolve_path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan: {data_path}")

    DEFAULT_YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "resume": args.resume,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_model_path.exists():
        print(f"Best model: {best_model_path}")

    if args.validate:
        model = YOLO(str(best_model_path if best_model_path.exists() else args.model))
        model.val(data=str(data_path), imgsz=args.imgsz, batch=args.batch, workers=args.workers)

    if args.test:
        model = YOLO(str(best_model_path if best_model_path.exists() else args.model))
        model.val(
            data=str(data_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
        )

    return results


if __name__ == "__main__":
    main()
