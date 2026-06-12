# model_adapters.py
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "Ultralytics"))

import cv2
from ultralytics import YOLO

from config import (
    MODEL_1_IMAGE_SIZE,
    MODEL_1_CONFIDENCE,
    MODEL_2_DEFAULT_BEARING_TYPE,
    REQUIRED_ASSEMBLY_CLASSES,
)
from measurement import DEFAULT_TOLERANCE_MM, detect_bearing_circles


def normalize_detection_class(class_name):
    return str(class_name or "").strip().lower().replace("-", "_").replace(" ", "_")


def display_detection_class(class_name):
    normalized = normalize_detection_class(class_name)
    if normalized in {"not_good", "ng", "not_ok", "reject", "rejected"}:
        return "Reject"
    if normalized in {"pass", "ok"}:
        return "Pass"
    return str(class_name or "").strip()


def clamp_bbox(bbox, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


class AssemblyDetectionModel:
    """Adapter Model 1: deteksi kelengkapan assembly dari frame asli."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path) if os.path.exists(model_path) else None
        self.names = self.model.names if self.model else {}

    def predict(self, frame, confidence=None, image_size=None):
        if self.model is None:
            return {
                "status_kelengkapan": "Tidak Lengkap",
                "status": "REJECT",
                "objects": [],
                "roi_bbox": None,
                "message": f"Model 1 belum tersedia: {self.model_path}",
            }

        results = self.model.predict(
            frame,
            conf=MODEL_1_CONFIDENCE if confidence is None else confidence,
            imgsz=MODEL_1_IMAGE_SIZE if image_size is None else image_size,
            verbose=False,
        )
        objects = []

        for result in results:
            if not result.boxes:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                confidence = float(box.conf[0])
                bbox = clamp_bbox(box.xyxy[0].tolist(), frame.shape)
                if bbox is None:
                    continue
                objects.append(
                    {
                        "label": display_detection_class(label),
                        "normalized_label": normalize_detection_class(label),
                        "raw_label": label,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    }
                )

        status_kelengkapan = self._status_kelengkapan(objects)
        return {
            "status_kelengkapan": status_kelengkapan,
            "status": "PASS" if status_kelengkapan == "Lengkap" else "REJECT",
            "objects": objects,
            "roi_bbox": self._select_roi_bbox(objects),
        }

    def _status_kelengkapan(self, objects):
        labels = {item["normalized_label"] for item in objects}
        required = {normalize_detection_class(item) for item in REQUIRED_ASSEMBLY_CLASSES}

        pass_score = max(
            [item["confidence"] for item in objects if item["normalized_label"] in {"pass", "ok"}],
            default=0,
        )
        reject_score = max(
            [
                item["confidence"]
                for item in objects
                if item["normalized_label"] in {"not_good", "ng", "not_ok", "reject", "rejected"}
            ],
            default=0,
        )

        if pass_score or reject_score:
            return "Lengkap" if pass_score >= reject_score else "Tidak Lengkap"

        if required:
            return "Lengkap" if required.issubset(labels) else "Tidak Lengkap"

        has_valid_object = any(label != "no_bearing" for label in labels)
        return "Lengkap" if has_valid_object else "Tidak Lengkap"

    def _select_roi_bbox(self, objects):
        candidates = [
            item
            for item in objects
            if item["normalized_label"] != "no_bearing"
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]),
        )["bbox"]


class BearingInfoModel:
    """
    Adapter Model 2: identifikasi diameter dan jenis bearing dari crop ROI.

    Ganti isi predict() dengan model diameter/jenis asli jika checkpoint sudah
    tersedia. Kontraknya tetap sama: input crop BGR, output diameter, jenis,
    status OK/NG.
    """

    def predict(self, crop, class_hint=None):
        if crop is None or crop.size == 0:
            return {
                "diameter": None,
                "jenis_bearing": None,
                "status": "NG",
                "reason": "ROI bearing tidak tersedia",
            }

        height, width = crop.shape[:2]
        # Alur utama: YOLO menentukan kelas ukuran dulu, lalu OpenCV mengukur
        # diameter berdasarkan standar dari kelas tersebut.
        bearing_type_hint = class_hint or MODEL_2_DEFAULT_BEARING_TYPE
        _, measurement = detect_bearing_circles(
            crop.copy(),
            [0, 0, width, height],
            bearing_type_hint,
            tolerance=DEFAULT_TOLERANCE_MM,
            draw=False,
        )

        outer_diameter = measurement.get("outer_diameter_mm")
        outer_diameter_px = measurement.get("outer_diameter_px")
        bearing_type = measurement.get("class") or bearing_type_hint
        status = "OK" if measurement.get("status") == "PASS" else "NG"

        return {
            "diameter": self._format_diameter(outer_diameter, outer_diameter_px),
            "jenis_bearing": bearing_type,
            "status": status,
            "measurement": measurement,
        }

    def _format_diameter(self, outer_diameter_mm, outer_diameter_px):
        if outer_diameter_mm is not None:
            return f"{outer_diameter_mm:.2f} mm"
        if outer_diameter_px is not None:
            return f"{outer_diameter_px} px"
        return None


def crop_from_frame(frame, bbox):
    clean_bbox = clamp_bbox(bbox, frame.shape) if bbox else None
    if clean_bbox is None:
        return None
    x1, y1, x2, y2 = clean_bbox
    return frame[y1:y2, x1:x2].copy()


def draw_final_result(frame, result):
    kelengkapan = result.get("kelengkapan", {})
    bearing_info = result.get("bearing_info", {})
    final_status = result.get("final_status", "REJECT")
    final_color = (0, 180, 0) if final_status == "PASS" else (0, 0, 255)

    for item in kelengkapan.get("objects", []):
        x1, y1, x2, y2 = item["bbox"]
        label = item["label"]
        confidence = item["confidence"]
        item_bearing_info = item.get("bearing_info", {})
        diameter = item_bearing_info.get("diameter")
        bearing_type = item_bearing_info.get("jenis_bearing")
        cv2.rectangle(frame, (x1, y1), (x2, y2), final_color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            final_color,
            2,
        )
        if diameter or bearing_type:
            cv2.putText(
                frame,
                f"{bearing_type or '-'} {diameter or '-'}",
                (x1, min(frame.shape[0] - 8, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                final_color,
                2,
            )

    lines = [
        f"Kelengkapan: {kelengkapan.get('status_kelengkapan', '-')}",
        f"Diameter: {bearing_info.get('diameter') or '-'}",
        f"Jenis: {bearing_info.get('jenis_bearing') or '-'}",
        f"Final: {final_status}",
    ]
    x, y = 10, 30
    cv2.rectangle(frame, (5, 5), (360, 120), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            final_color if index == len(lines) - 1 else (255, 255, 255),
            2,
        )

    return frame
