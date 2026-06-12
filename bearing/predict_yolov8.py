import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict gambar/folder memakai model YOLOv8 dari folder runs."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path gambar atau folder gambar yang ingin dipredict.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/bearing_yolov8/weights/best.pt"),
        help="Path model YOLOv8 .pt.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Ukuran input image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold untuk NMS.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device inference. Contoh: cpu atau 0. Default auto.",
    )
    parser.add_argument(
        "--output",
        default="runs/predict",
        help="Folder output hasil predict.",
    )
    parser.add_argument(
        "--name",
        default="bearing_predict",
        help="Nama folder run predict.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Tampilkan hasil dalam format JSON.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Simpan label prediksi YOLO txt.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Tampilkan window visual prediksi secara langsung saat inference.",
    )
    parser.add_argument(
        "--open-output",
        action="store_true",
        help="Buka gambar hasil prediksi pertama setelah inference selesai.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_images(source):
    if source.is_file():
        return [source]

    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def box_to_dict(box, names):
    class_id = int(box.cls.item())
    xyxy = [round(float(value), 2) for value in box.xyxy.squeeze(0).tolist()]
    xywhn = [round(float(value), 6) for value in box.xywhn.squeeze(0).tolist()]

    return {
        "class_id": class_id,
        "class_name": names[class_id],
        "confidence": round(float(box.conf.item()), 6),
        "bbox_xyxy": {
            "x1": xyxy[0],
            "y1": xyxy[1],
            "x2": xyxy[2],
            "y2": xyxy[3],
        },
        "bbox_yolo": {
            "x_center": xywhn[0],
            "y_center": xywhn[1],
            "width": xywhn[2],
            "height": xywhn[3],
        },
    }


def summarize_results(results, names):
    summaries = []
    for result in results:
        detections = [box_to_dict(box, names) for box in result.boxes]
        summaries.append(
            {
                "image": str(Path(result.path)),
                "detections": detections,
            }
        )
    return summaries


def print_text_summary(summaries, output_dir):
    print(f"Output: {output_dir}")
    print()

    for item in summaries:
        print(f"Gambar: {item['image']}")

        if not item["detections"]:
            print("  Deteksi: tidak ada")
            print()
            continue

        for detection in item["detections"]:
            bbox = detection["bbox_xyxy"]
            print(
                "  - {class_name}: conf={confidence:.4f}, "
                "box=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})".format(
                    class_name=detection["class_name"],
                    confidence=detection["confidence"],
                    x1=bbox["x1"],
                    y1=bbox["y1"],
                    x2=bbox["x2"],
                    y2=bbox["y2"],
                )
            )
        print()


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def find_output_images(output_dir):
    if not output_dir.exists():
        return []

    return sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main():
    args = parse_args()
    source = resolve_path(args.source)
    model_path = resolve_path(args.model)

    if not source.exists():
        raise FileNotFoundError(f"Source tidak ditemukan: {source}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    image_paths = collect_images(source)
    if not image_paths:
        raise FileNotFoundError(f"Tidak ada gambar ditemukan di: {source}")

    model = YOLO(str(model_path))
    predict_kwargs = {
        "source": str(source),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "project": args.output,
        "name": args.name,
        "save": True,
        "save_txt": args.save_txt,
        "show": args.show,
        "exist_ok": True,
    }

    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    output_dir = Path(args.output) / args.name
    summaries = summarize_results(results, model.names)

    if args.json:
        print(json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2))
    else:
        print_text_summary(summaries, output_dir)

    if args.open_output:
        output_images = find_output_images(output_dir)
        if output_images:
            open_file(str(output_images[0]))
        else:
            print(f"Tidak ada gambar output untuk dibuka di: {output_dir}")


if __name__ == "__main__":
    main()
