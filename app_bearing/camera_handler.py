# camera_handler.py
import cv2
import base64
import threading
import time
import numpy as np
from collections import Counter, defaultdict, deque
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "Ultralytics"))

from ultralytics import YOLO
from config import (
    ASSEMBLY_STABILITY_FRAMES,
    ASSEMBLY_PASS_BY_DETECTED_BEARINGS,
    BIG_BEARING_MIN_BBOX_WIDTH,
    COLORS,
    CLASS_STATUS,
    DAILY_TARGET,
    DETECTION_CONFIDENCE,
    DETECTION_IMAGE_SIZE,
    DETECTION_CLASS_STABILITY_FRAMES,
    DETECTION_STABILITY_FRAMES,
    DETECTION_DEDUP_IOU_THRESHOLD,
    LIVE_INSPECTION_INTERVAL,
    MAX_BEARING_DETECTIONS,
    REQUIRED_BEARING_COUNT,
    SIMILAR_BBOX_AREA_RATIO,
    STABILITY_LOCK_COUNT,
    USE_BBOX_SIZE_CLASS_CORRECTION,
    USE_BEARING_FALLBACK_FOR_MODEL1_MISSING,
    USE_BEARING_FALLBACK_WHEN_MODEL1_EMPTY,
    SHOW_MODEL1_BOX_WHEN_BEARINGS_DECIDE,
    USE_TWO_BEARING_SLOT_CLASS_CORRECTION,
    USE_MODEL1_FOR_ASSEMBLY_DECISION,
)
from model_adapters import AssemblyDetectionModel
from measurement import DEFAULT_TOLERANCE_MM, detect_bearing_circles

def normalize_detection_class(class_name):
    return str(class_name or '').strip().lower().replace('-', '_').replace(' ', '_')


def bearing_class_rank(class_name):
    normalized = normalize_detection_class(class_name)
    if normalized in {'big_bearings', 'big_bearing', 'large_bearing', 'large', '6301z'}:
        return 3
    if normalized in {'medium_bearing', 'medium', '608z'}:
        return 2
    if normalized in {'small_bearing', 'small', '688z'}:
        return 1
    return 0

class CameraHandler:
    def __init__(self, model_path, conveyor, inspection, assembly_model_path=None):
        self.model = YOLO(model_path)
        self.assembly_model = AssemblyDetectionModel(assembly_model_path) if assembly_model_path else None
        self.conveyor = conveyor
        self.inspection = inspection
        self.settings = {
            "threshold": DETECTION_CONFIDENCE,
            "assembly_threshold": 0.2,
            "zoom": 1.0,
            "brightness": 0,
            "noise": 0,
            "big_min_bbox_width": BIG_BEARING_MIN_BBOX_WIDTH,
        }
        self.camera = None
        self.active = False
        self.callback = None
        self.assembly_status_history = deque(maxlen=ASSEMBLY_STABILITY_FRAMES)
        self.detection_status_history = defaultdict(lambda: deque(maxlen=DETECTION_STABILITY_FRAMES))
        self.detection_class_history = defaultdict(lambda: deque(maxlen=DETECTION_CLASS_STABILITY_FRAMES))
        self.locked_assembly_status = None
        self.locked_detection_status = {}
        self.locked_detection_class = {}

    def update_settings(self, settings):
        numeric_ranges = {
            "threshold": (0.05, 0.95),
            "assembly_threshold": (0.05, 0.95),
            "zoom": (1.0, 3.0),
            "brightness": (-80, 80),
            "noise": (0, 3),
            "big_min_bbox_width": (60, 320),
        }
        updated = {}
        for key, value in (settings or {}).items():
            if key not in numeric_ranges:
                continue
            minimum, maximum = numeric_ranges[key]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            numeric_value = max(minimum, min(maximum, numeric_value))
            if key == "noise":
                numeric_value = int(round(numeric_value))
            self.settings[key] = numeric_value
            updated[key] = numeric_value
        return self.get_settings()

    def get_settings(self):
        return dict(self.settings)
    
    def start(self, camera_index=1, callback=None):
        self.callback = callback
        if self.camera:
            self.camera.release()
        
        try:
            self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(camera_index)
            
            if self.camera.isOpened():
                self.active = True
                thread = threading.Thread(target=self._stream, daemon=True)
                thread.start()
                return {'status': 'started', 'index': camera_index}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
        
        return {'status': 'error', 'message': f'Camera {camera_index} not found'}
    
    def stop(self):
        self.active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.assembly_status_history.clear()
        self.detection_status_history.clear()
        self.detection_class_history.clear()
        self.locked_assembly_status = None
        self.locked_detection_status.clear()
        self.locked_detection_class.clear()
        return {'status': 'stopped'}

    def _apply_frame_settings(self, frame):
        settings = self.settings
        processed = frame

        zoom = float(settings.get("zoom", 1.0))
        if zoom > 1.01:
            height, width = processed.shape[:2]
            crop_width = max(1, int(width / zoom))
            crop_height = max(1, int(height / zoom))
            x1 = (width - crop_width) // 2
            y1 = (height - crop_height) // 2
            processed = processed[y1:y1 + crop_height, x1:x1 + crop_width]
            processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_LINEAR)

        brightness = int(settings.get("brightness", 0))
        if brightness:
            processed = cv2.convertScaleAbs(processed, alpha=1.0, beta=brightness)

        noise = int(settings.get("noise", 0))
        if noise == 1:
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
        elif noise == 2:
            processed = cv2.medianBlur(processed, 3)
        elif noise >= 3:
            processed = cv2.bilateralFilter(processed, 7, 50, 50)

        return processed

    def _predict_assembly(self, frame):
        if not self.assembly_model:
            return {
                'status_kelengkapan': 'Tidak Tersedia',
                'status': 'REJECT',
                'objects': [],
                'roi_bbox': None,
                'message': 'Model kelengkapan belum dikonfigurasi'
            }
        return self.assembly_model.predict(
            frame,
            confidence=self.settings.get("assembly_threshold"),
        )

    def _lock_value(self, history, raw_value, current_lock=None):
        history.append(raw_value)
        value, count = Counter(history).most_common(1)[0]
        if len(history) == history.maxlen and count >= STABILITY_LOCK_COUNT:
            return value
        return current_lock if current_lock is not None else raw_value

    def _stabilize_assembly(self, assembly):
        status = assembly.get('status', 'REJECT')
        stable_status = self._lock_value(
            self.assembly_status_history,
            status,
            self.locked_assembly_status,
        )
        self.locked_assembly_status = stable_status
        assembly['raw_status'] = status
        assembly['status'] = stable_status
        assembly['status_kelengkapan'] = 'Lengkap' if stable_status == 'PASS' else 'Tidak Lengkap'
        return assembly

    def _stabilize_detections(self, detections):
        occurrence = defaultdict(int)
        for detection in sorted(detections, key=lambda item: (item.get('bbox') or [0])[0]):
            measurement = detection.get('measurement') or {}
            class_key = measurement.get('class') or detection.get('normalized_class') or detection.get('class')
            occurrence[class_key] += 1
            key = f"{class_key}:{occurrence[class_key]}"
            status = detection.get('status', 'REJECT')
            history = self.detection_status_history[key]
            stable_status = self._lock_value(
                history,
                status,
                self.locked_detection_status.get(key),
            )
            self.locked_detection_status[key] = stable_status
            detection['raw_status'] = status
            detection['status'] = stable_status
            if measurement:
                measurement['raw_status'] = measurement.get('status', status)
                measurement['status'] = stable_status
                if stable_status == 'PASS' and measurement.get('reason') not in {'Measurement OK', 'ID not detected, accepted by OD'}:
                    measurement['reason'] = 'Stable by frame voting'
        return detections

    def _bbox_area(self, detection):
        x1, y1, x2, y2 = detection['bbox']
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _deduplicate_detections(self, detections):
        if not detections:
            return detections

        kept = []
        for detection in sorted(detections, key=lambda item: item.get('confidence', 0), reverse=True):
            bbox = detection.get('bbox')
            if not bbox:
                continue
            duplicate = any(
                self._bbox_iou(bbox, other.get('bbox', [0, 0, 0, 0])) >= DETECTION_DEDUP_IOU_THRESHOLD
                for other in kept
            )
            if not duplicate:
                kept.append(detection)

        if USE_BBOX_SIZE_CLASS_CORRECTION and len(kept) > MAX_BEARING_DETECTIONS:
            by_area = sorted(kept, key=self._bbox_area)
            if MAX_BEARING_DETECTIONS == 3:
                middle_index = len(by_area) // 2 - 1 if len(by_area) % 2 == 0 else len(by_area) // 2
                selected = [by_area[0], by_area[middle_index], by_area[-1]]
                unique = []
                seen = set()
                for item in selected:
                    item_id = id(item)
                    if item_id not in seen:
                        seen.add(item_id)
                        unique.append(item)
                kept = unique
            else:
                kept = by_area[:MAX_BEARING_DETECTIONS]

        return sorted(kept, key=lambda item: (item.get('bbox') or [0])[0])

    def _bbox_iou(self, first, second):
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection <= 0:
            return 0
        first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = first_area + second_area - intersection
        return intersection / union if union else 0

    def _stabilize_detection_classes(self, detections):
        for detection in detections:
            bbox = detection.get('bbox') or [0, 0, 0, 0]
            bbox_width = max(0, bbox[2] - bbox[0])
            if bbox_width >= self.settings.get("big_min_bbox_width", BIG_BEARING_MIN_BBOX_WIDTH):
                detection['raw_class'] = detection.get('class')
                detection['raw_normalized_class'] = detection.get('normalized_class')
                detection['class'] = 'big_bearings'
                detection['normalized_class'] = normalize_detection_class('big_bearings')
                detection['class_source'] = 'bbox_width'

        if USE_TWO_BEARING_SLOT_CLASS_CORRECTION and len(detections) == 2:
            by_slot = sorted(
                detections,
                key=lambda item: ((item.get('bbox') or [0, 0, 0, 0])[0] + (item.get('bbox') or [0, 0, 0, 0])[2]) / 2,
            )
            selected = [by_slot[0], by_slot[-1]]
            first_area = self._bbox_area(selected[0])
            second_area = self._bbox_area(selected[1])
            larger_area = max(first_area, second_area)
            area_delta_ratio = abs(first_area - second_area) / larger_area if larger_area else 1
            if area_delta_ratio <= SIMILAR_BBOX_AREA_RATIO:
                reference = max(
                    selected,
                    key=lambda item: (bearing_class_rank(item.get('class')), item.get('confidence', 0)),
                )
                fixed_class = reference['class']
                fixed_normalized_class = reference['normalized_class']
                for detection in selected:
                    detection['raw_class'] = detection['class']
                    detection['raw_normalized_class'] = detection['normalized_class']
                    detection['class'] = fixed_class
                    detection['normalized_class'] = fixed_normalized_class
                    detection['class_source'] = 'similar_area'
            return selected

        if USE_BBOX_SIZE_CLASS_CORRECTION and len(detections) >= 2:
            detections = sorted(detections, key=self._bbox_area)[:MAX_BEARING_DETECTIONS]
            if len(detections) >= REQUIRED_BEARING_COUNT:
                ordered_detections = sorted(detections, key=self._bbox_area)
                fixed_classes = ['small_bearing', 'medium_bearing', 'big_bearings']
                class_source = 'bbox_size'
            else:
                # Untuk mode 2 bearing pada jig saat ini, slot kiri adalah 608Z
                # dan slot kanan adalah 6301Z. Ini lebih stabil daripada area
                # bbox karena perspektif kamera bisa membuat ukuran tampak mirip.
                ordered_detections = sorted(
                    detections,
                    key=lambda item: ((item.get('bbox') or [0, 0, 0, 0])[0] + (item.get('bbox') or [0, 0, 0, 0])[2]) / 2,
                )
                fixed_classes = ['medium_bearing', 'big_bearings']
                class_source = 'slot_position'
            for detection, fixed_class in zip(ordered_detections, fixed_classes):
                detection['raw_class'] = detection['class']
                detection['raw_normalized_class'] = detection['normalized_class']
                detection['class'] = fixed_class
                detection['normalized_class'] = normalize_detection_class(fixed_class)
                detection['class_source'] = class_source
            return detections

        occurrence = defaultdict(int)
        for detection in sorted(detections, key=lambda item: (item.get('bbox') or [0])[0]):
            occurrence['slot'] += 1
            key = f"slot:{occurrence['slot']}"
            raw_class = detection['class']
            history = self.detection_class_history[key]
            stable_class = self._lock_value(
                history,
                raw_class,
                self.locked_detection_class.get(key),
            )
            self.locked_detection_class[key] = stable_class
            detection['raw_class'] = raw_class
            detection['class'] = stable_class
            detection['normalized_class'] = normalize_detection_class(stable_class)
            detection['class_source'] = 'frame_vote' if stable_class != raw_class else 'model'
        return detections

    def _assembly_from_bearings(self, assembly, detections):
        model1_has_reading = bool(assembly.get('objects'))
        if (
            USE_MODEL1_FOR_ASSEMBLY_DECISION
            and model1_has_reading
            and USE_BEARING_FALLBACK_FOR_MODEL1_MISSING
        ):
            self._fill_model1_missing_bearings(assembly, detections)
            labels = {normalize_detection_class(item.get('label')) for item in assembly.get('objects', [])}
            has_reject = bool(labels & {'not_good', 'ng', 'not_ok', 'reject', 'rejected'})
            fallback_count = sum(1 for item in assembly.get('objects', []) if item.get('source') == 'bearing_fallback')
            if not has_reject and len(detections) >= REQUIRED_BEARING_COUNT and fallback_count:
                assembly['status'] = 'PASS'
                assembly['status_kelengkapan'] = 'Lengkap'
                assembly['message'] = 'Model 1 dilengkapi dari deteksi bearing'
                assembly['decision_source'] = 'model_1_with_bearing_fallback'
                return assembly

        if USE_MODEL1_FOR_ASSEMBLY_DECISION and model1_has_reading:
            assembly['decision_source'] = 'model_1'
            return assembly

        if (
            USE_MODEL1_FOR_ASSEMBLY_DECISION
            and not model1_has_reading
            and not USE_BEARING_FALLBACK_WHEN_MODEL1_EMPTY
        ):
            assembly['decision_source'] = 'model_1'
            return assembly

        if (
            USE_MODEL1_FOR_ASSEMBLY_DECISION
            and not model1_has_reading
            and USE_BEARING_FALLBACK_WHEN_MODEL1_EMPTY
            and detections
        ):
            self._fill_model1_missing_bearings(assembly, detections)
            has_reject = any(detection.get('status') == 'REJECT' for detection in detections)
            assembly['raw_status'] = assembly.get('status', 'REJECT')
            assembly['raw_status_kelengkapan'] = assembly.get('status_kelengkapan', 'Tidak Lengkap')
            assembly['status'] = 'REJECT' if has_reject else 'PASS'
            assembly['status_kelengkapan'] = 'Tidak Lengkap' if has_reject else 'Lengkap'
            assembly['message'] = 'Model 1 belum membaca; memakai deteksi bearing'
            assembly['decision_source'] = 'model_1_empty_bearing_fallback'
            return assembly

        if not ASSEMBLY_PASS_BY_DETECTED_BEARINGS:
            return assembly

        detected_types = {
            (detection.get('measurement') or {}).get('class')
            for detection in detections
        }
        detected_types.discard(None)
        required_types = {'688Z', '608Z', '6301Z'}
        has_required_types = required_types.issubset(detected_types)
        has_required_count = len(detections) >= REQUIRED_BEARING_COUNT

        if has_required_types or has_required_count:
            assembly['raw_status'] = assembly.get('status', 'REJECT')
            assembly['raw_status_kelengkapan'] = assembly.get('status_kelengkapan', 'Tidak Lengkap')
            assembly['status'] = 'PASS'
            assembly['status_kelengkapan'] = 'Lengkap'
            assembly['message'] = 'Kelengkapan dari 3 bearing terdeteksi'
            assembly['decision_source'] = 'detected_bearings'
        else:
            if USE_MODEL1_FOR_ASSEMBLY_DECISION:
                assembly['decision_source'] = 'model_1'
            else:
                assembly['raw_status'] = assembly.get('status', 'REJECT')
                assembly['raw_status_kelengkapan'] = assembly.get('status_kelengkapan', 'Tidak Lengkap')
                assembly['status'] = 'REJECT'
                assembly['status_kelengkapan'] = 'Tidak Lengkap'
                assembly['message'] = f'Bearing terdeteksi {len(detections)}/{REQUIRED_BEARING_COUNT}'
                assembly['decision_source'] = 'detected_bearings'
        return assembly

    def _fill_model1_missing_bearings(self, assembly, detections):
        objects = assembly.setdefault('objects', [])
        for detection in detections:
            bbox = detection.get('bbox')
            if not bbox:
                continue
            has_overlap = any(self._bbox_iou(bbox, item.get('bbox', [0, 0, 0, 0])) >= 0.15 for item in objects)
            if has_overlap:
                continue
            measurement = detection.get('measurement') or {}
            label = f"Pass {measurement.get('class') or detection.get('class', '')}".strip()
            objects.append({
                'label': label,
                'normalized_label': 'pass',
                'confidence': detection.get('confidence', 0) / 100,
                'bbox': bbox,
                'source': 'bearing_fallback',
            })

    def _measure_detections(self, frame, detections):
        for detection in detections:
            _, measurement = detect_bearing_circles(
                frame,
                detection['bbox'],
                detection['class'],
                tolerance=DEFAULT_TOLERANCE_MM,
                draw=False
            )
            detection['measurement'] = measurement
            detection['status'] = measurement.get('status', 'REJECT')
        return detections

    def _draw_assembly_result(self, frame, assembly):
        status = assembly.get('status', 'REJECT')
        status_kelengkapan = assembly.get('status_kelengkapan', '-')
        color = (0, 200, 0) if status == 'PASS' else (0, 0, 255)
        wanted_labels = {'pass', 'ok'} if status == 'PASS' else {'not_good', 'ng', 'not_ok', 'reject', 'rejected'}
        if assembly.get('decision_source') == 'detected_bearings' and not SHOW_MODEL1_BOX_WHEN_BEARINGS_DECIDE:
            display_objects = []
        else:
            display_objects = [
                item for item in assembly.get('objects', [])
                if normalize_detection_class(item.get('label')) in wanted_labels
            ] or assembly.get('objects', [])

        for item in display_objects:
            x1, y1, x2, y2 = item['bbox']
            item_color = color
            cv2.rectangle(frame, (x1, y1), (x2, y2), item_color, 3)

        panel_lines = [
            f"Model 1: {status_kelengkapan}",
            f"Final: {status}",
        ]
        if assembly.get('decision_source') == 'detected_bearings':
            panel_lines[0] = f"Kelengkapan: {status_kelengkapan}"
        cv2.rectangle(frame, (10, frame.shape[0] - 70), (330, frame.shape[0] - 10), (0, 0, 0), -1)
        for index, text in enumerate(panel_lines):
            cv2.putText(
                frame,
                text,
                (20, frame.shape[0] - 45 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )
        return frame

    def _draw_detection_result(self, yolo_frame, opencv_frame, detection):
        x1, y1, x2, y2 = detection['bbox']
        measurement = detection.get('measurement') or {}
        status = detection.get('status', measurement.get('status', 'REJECT'))
        color = (0, 200, 0) if status == 'PASS' else (0, 0, 255)
        class_name = detection.get('class', '-')
        normalized_class = detection.get('normalized_class', class_name)
        bearing_type = measurement.get('class', class_name)
        confidence = detection.get('confidence', 0)
        display_index = detection.get('display_index', 0)

        yolo_color = COLORS.get(class_name, COLORS.get(normalized_class, color))
        cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), yolo_color, 3)
        label = f"#{display_index} {bearing_type}"
        confidence_label = f"{confidence:.0f}% {status}"
        label_y = y1 - 25 if y1 >= 42 else y2 + 18
        cv2.putText(yolo_frame, label, (x1 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolo_color, 2)
        cv2.putText(yolo_frame, confidence_label, (x1 + 4, label_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, yolo_color, 2)

        cv2.rectangle(opencv_frame, (x1, y1), (x2, y2), color, 2)
        marker = f"#{display_index}"
        cv2.putText(opencv_frame, marker, (x1 + 5, max(17, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    def _draw_detection_summary_panel(self, frame, detections):
        if not detections:
            return

        panel_lines = []
        for detection in detections:
            measurement = detection.get('measurement') or {}
            bearing_type = measurement.get('class', detection.get('class', '-'))
            od = measurement.get('outer_diameter_mm')
            bearing_id = measurement.get('inner_diameter_mm')
            status = detection.get('status', measurement.get('status', '-'))
            od_text = f"{od:.2f}" if od is not None else "N/A"
            id_text = f"{bearing_id:.2f}" if bearing_id is not None else "N/A"
            panel_lines.append(f"#{detection.get('display_index', 0)} {bearing_type}  OD {od_text}  ID {id_text}  {status}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.46
        thickness = 1
        line_height = 19
        panel_width = 0
        for line in panel_lines:
            size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            panel_width = max(panel_width, size[0])

        x = 10
        y = frame.shape[0] - (line_height * len(panel_lines)) - 12
        y = max(92, y)
        panel_height = line_height * len(panel_lines) + 10
        cv2.rectangle(frame, (x - 5, y - 16), (x + panel_width + 12, y + panel_height - 16), (0, 0, 0), -1)
        for index, line in enumerate(panel_lines):
            cv2.putText(frame, line, (x, y + index * line_height), font, font_scale, (0, 255, 0), thickness + 1)
    
    def _stream(self):
        locked_latency = 0.0
        while self.active and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                break
            
            start_time = time.perf_counter()

            frame = self._apply_frame_settings(frame)
            
            raw_assembly = self._predict_assembly(frame)
            results = self.model.predict(
                frame,
                conf=self.settings.get("threshold", DETECTION_CONFIDENCE),
                imgsz=DETECTION_IMAGE_SIZE,
                verbose=False,
            )
            yolo_frame = frame.copy()
            opencv_frame = frame.copy()
            detections = []
            
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        normalized_class = normalize_detection_class(class_name)
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detections.append({
                            'class': class_name,
                            'normalized_class': normalized_class,
                            'confidence': round(confidence * 100, 1),
                            'bbox': [x1, y1, x2, y2],
                        })
            detections = self._deduplicate_detections(detections)
            detections = self._stabilize_detection_classes(detections)
            detections = self._measure_detections(frame, detections)
            detections = self._stabilize_detections(detections)
            assembly = self._assembly_from_bearings(raw_assembly, detections)
            assembly = self._stabilize_assembly(assembly)
            self._draw_assembly_result(yolo_frame, assembly)
            for index, detection in enumerate(detections, start=1):
                detection['display_index'] = index
                self._draw_detection_result(yolo_frame, opencv_frame, detection)
            self._draw_detection_summary_panel(opencv_frame, detections)

            final_reject = assembly.get('status') == 'REJECT'
            if detections:
                self.conveyor.add_production(len(detections))
                defect = sum(1 for d in detections if d.get('status') == 'REJECT')
                if defect > 0 or final_reject:
                    self.conveyor.add_defect(defect)
                    if self.conveyor.auto_stop_if_needed():
                        if self.callback:
                            pass
                    if self.callback:
                        pass
            
            stats = self.inspection.get_statistics()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            current_fps = 1000.0 / latency_ms if latency_ms > 0 else 0
            
            # Bekukan angka Latency saat ada bearing di layar agar mudah dicatat.
            # Ketika mesin berhenti (STOP), bearing masih di layar, jadi angka Latency 
            # akan tetap beku dan bisa kamu catat! FPS tetap dibiarkan berjalan (Live).
            if detections:
                if locked_latency == 0.0:
                    locked_latency = latency_ms
                    # Simpan hasil live detection ke database secara otomatis
                    # Ini mencegah spam DB karena hanya disimpan 1x per barang lewat
                    if self.inspection:
                        for d in detections:
                            self.inspection.add(
                                class_name=d.get('class', 'Unknown'),
                                confidence=d.get('confidence', 0.0),
                                image_name="Live Camera",
                                measurement=d.get('measurement'),
                                m1_confidence=assembly.get('assembly_confidence')
                            )
                display_latency = locked_latency
            else:
                locked_latency = 0.0
                display_latency = latency_ms
            
            assembly['latency_ms'] = display_latency
            
            overlay = [
                f"Production: {stats['total']} / {DAILY_TARGET}",
                f"Defect: {stats['defect_rate']:.1f}% | Latency: {display_latency:.1f} ms",
                f"Shift: {stats['shift']} | FPS: {current_fps:.1f}",
                f"Conveyor: {'RUN' if self.conveyor.status['running'] else 'STOP'} | Speed: {self.conveyor.status['speed']}%"
            ]
            for i, text in enumerate(overlay):
                y = 30 + i*25
                cv2.putText(yolo_frame, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(opencv_frame, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            color = (0,255,0) if self.conveyor.status['running'] else (0,0,255)
            cv2.rectangle(yolo_frame, (0,0), (yolo_frame.shape[1], 5), color, -1)
            cv2.rectangle(opencv_frame, (0,0), (opencv_frame.shape[1], 5), color, -1)
            
            _, yolo_buffer = cv2.imencode('.jpg', yolo_frame)
            _, opencv_buffer = cv2.imencode('.jpg', opencv_frame)
            yolo_base64 = base64.b64encode(yolo_buffer).decode('utf-8')
            opencv_base64 = base64.b64encode(opencv_buffer).decode('utf-8')
            
            if self.callback:
                self.callback(yolo_base64, detections, len(detections), opencv_base64, assembly)
            
            time.sleep(LIVE_INSPECTION_INTERVAL)
        
        if self.camera:
            self.camera.release()
    
    def detect_image(self, image_bytes):
        start_time = time.perf_counter()
        self._reset_stabilization()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = self._apply_frame_settings(img)
        
        raw_assembly = self._predict_assembly(img)
        results = self.model.predict(
            img,
            conf=self.settings.get("threshold", DETECTION_CONFIDENCE),
            imgsz=DETECTION_IMAGE_SIZE,
            verbose=False,
        )
        
        detections = []
        annotated = img.copy()
        
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    normalized_class = normalize_detection_class(class_name)
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        'class': class_name,
                        'normalized_class': normalized_class,
                        'confidence': round(confidence * 100, 1),
                        'bbox': [x1, y1, x2, y2],
                    })
        detections = self._deduplicate_detections(detections)
        detections = self._stabilize_detection_classes(detections)
        detections = self._measure_detections(img, detections)
        detections = self._stabilize_detections(detections)
        assembly = self._assembly_from_bearings(raw_assembly, detections)
        assembly = self._stabilize_assembly(assembly)
        self._draw_assembly_result(annotated, assembly)
        for index, detection in enumerate(detections, start=1):
            detection['display_index'] = index
            self._draw_detection_result(annotated, annotated, detection)
        self._draw_detection_summary_panel(annotated, detections)
        
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        assembly['latency_ms'] = (time.perf_counter() - start_time) * 1000
        
        return detections, img_base64, assembly

    def _reset_stabilization(self):
        self.assembly_status_history.clear()
        self.detection_status_history.clear()
        self.detection_class_history.clear()
        self.locked_assembly_status = None
        self.locked_detection_status.clear()
        self.locked_detection_class.clear()
