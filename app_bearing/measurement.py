"""
measurement.py

OpenCV measurement module for bearing quality control.

Workflow:
1. Use the YOLO bounding box as the Region of Interest (ROI).
2. Detect the outer and inner bearing circles inside the ROI.
3. Convert pixel diameters into millimeters using the detected outer circle
   and the known standard OD for the predicted bearing class.
4. Validate OD and ID against bearing standards with a configurable tolerance.
5. Draw the measurement result on the original frame.
"""

import cv2
import numpy as np

from config import USE_ID_FOR_VALIDATION


# Bearing dimensional standards in millimeters.
BEARING_STANDARDS = {
    "6301Z": {"od": 37.0, "id": 12.0},
    "608Z": {"od": 22.0, "id": 8.0},
    "688Z": {"od": 16.0, "id": 8.0},
}

DEFAULT_TOLERANCE_MM = 2.0
ID_TOLERANCE_MM = 3.0

CLASS_TO_BEARING_TYPE = {
    "SMALL": "688Z",
    "SMALL_BEARING": "688Z",
    "SMALL BEARING": "688Z",
    "MEDIUM": "608Z",
    "MEDIUM_BEARING": "608Z",
    "MEDIUM BEARING": "608Z",
    "LARGE": "6301Z",
    "LARGE_BEARING": "6301Z",
    "LARGE BEARING": "6301Z",
    "BIG": "6301Z",
    "BIG_BEARING": "6301Z",
    "BIG_BEARINGS": "6301Z",
    "BIG BEARING": "6301Z",
    "BIG BEARINGS": "6301Z",
}


def _normalize_class_name(class_name):
    """Return a class name that matches the bearing standard keys."""
    if class_name is None:
        return ""
    class_name = str(class_name).strip().upper()
    class_name = class_name.replace("-", "_")
    return CLASS_TO_BEARING_TYPE.get(class_name, class_name)


def _clamp_bbox(bbox, frame_shape):
    """Keep YOLO bbox coordinates inside the image boundary."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _preprocess_roi(roi):
    """Convert ROI to grayscale and reduce noise before circle detection."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE improves contrast on metallic bearing surfaces with uneven light.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Median blur is stable for circular edge detection because it suppresses
    # salt-and-pepper sensor noise while preserving strong edges.
    blurred = cv2.medianBlur(gray, 5)
    return gray, blurred


def _detect_circles_hough(blurred):
    """Detect circle candidates with Hough transform."""
    height, width = blurred.shape[:2]
    min_side = min(width, height)

    if min_side < 20:
        return []

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min_side // 4),
        param1=100,
        param2=22,
        minRadius=max(4, int(min_side * 0.12)),
        maxRadius=max(6, int(min_side * 0.55)),
    )

    if circles is None:
        return []

    return np.round(circles[0]).astype(int).tolist()


def _detect_circles_contour(gray):
    """Fallback circle detection using contours and minimum enclosing circles."""
    height, width = gray.shape[:2]
    min_side = min(width, height)

    # Otsu threshold adapts to bright/dark bearing images better than one fixed
    # threshold. Morphology closes small gaps in circular edges.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    edges = cv2.Canny(binary, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # RETR_LIST keeps inner ring contours too; RETR_EXTERNAL would only keep
    # the largest outside boundary and can miss the bearing bore.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(contour)

        if (
            circularity >= 0.55
            and min_side * 0.08 <= radius <= min_side * 0.58
            and area >= 40
        ):
            circles.append([int(round(x)), int(round(y)), int(round(radius))])

    return circles


def _select_outer_inner_circles(circles, roi_shape):
    """Choose outer and inner circles from detected candidates."""
    if not circles:
        return None, None

    height, width = roi_shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0

    # Sort by radius descending. The biggest reliable circle is treated as OD.
    circles = sorted(circles, key=lambda item: item[2], reverse=True)
    outer = circles[0]
    outer_radius = outer[2]

    inner_candidates = []
    for circle in circles[1:]:
        x, y, radius = circle
        distance_to_outer = np.hypot(x - outer[0], y - outer[1])
        distance_to_roi_center = np.hypot(x - center_x, y - center_y)

        # Inner circle should be smaller and roughly concentric.
        if radius < outer_radius * 0.75 and distance_to_outer <= outer_radius * 0.35:
            inner_candidates.append((distance_to_roi_center, circle))

    if inner_candidates:
        inner = sorted(inner_candidates, key=lambda item: item[0])[0][1]
    else:
        inner = None

    return outer, inner


def _draw_text_panel(frame, lines, x, y, color):
    """Draw readable OpenCV status text near the detected ROI."""
    if not lines:
        return

    height, width = frame.shape[:2]
    x = max(5, min(int(x), width - 10))
    y = max(25, min(int(y), height - 10))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_height = 22
    panel_width = 0

    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        panel_width = max(panel_width, text_size[0])

    panel_height = line_height * len(lines) + 10
    if x + panel_width + 16 > width:
        x = max(5, width - panel_width - 16)
    if y + panel_height > height:
        y = max(25, height - panel_height)

    cv2.rectangle(
        frame,
        (x - 5, y - 18),
        (x + panel_width + 10, y + panel_height - 18),
        (0, 0, 0),
        -1,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            font,
            font_scale,
            color,
            thickness,
        )


def calculate_pixel_to_mm(outer_diameter_px, bearing_class):
    """
    Calculate millimeter scale from the detected OD pixel diameter.

    In this setup the known bearing OD is used as the calibration reference.
    For higher accuracy in production, replace this with camera calibration or
    a fixed px/mm value from a calibrated jig at the same camera height.
    """
    class_name = _normalize_class_name(bearing_class)
    standard = BEARING_STANDARDS.get(class_name)

    if not standard or outer_diameter_px <= 0:
        return None

    return standard["od"] / float(outer_diameter_px)


def validate_bearing_size(
    bearing_class,
    outer_diameter_mm,
    inner_diameter_mm,
    tolerance=DEFAULT_TOLERANCE_MM,
    id_tolerance=ID_TOLERANCE_MM,
):
    """Validate measured OD and ID against the bearing dimensional standard."""
    class_name = _normalize_class_name(bearing_class)
    standard = BEARING_STANDARDS.get(class_name)

    if not standard:
        return {
            "class": class_name,
            "outer_diameter_mm": outer_diameter_mm,
            "inner_diameter_mm": inner_diameter_mm,
            "status": "REJECT",
            "reason": "Unknown bearing class",
        }

    od_error = abs(outer_diameter_mm - standard["od"])
    id_error = (
        abs(inner_diameter_mm - standard["id"])
        if inner_diameter_mm is not None
        else None
    )

    od_pass = od_error <= tolerance
    if USE_ID_FOR_VALIDATION:
        id_pass = inner_diameter_mm is None or id_error <= id_tolerance
    else:
        id_pass = True
    status = "PASS" if od_pass and id_pass else "REJECT"

    reason = "Measurement OK"
    if not USE_ID_FOR_VALIDATION and od_pass:
        reason = "OD valid, ID ignored"
    elif inner_diameter_mm is None and od_pass:
        reason = "ID not detected, accepted by OD"
    elif not od_pass:
        reason = "OD outside tolerance"
    elif not id_pass:
        reason = "ID outside tolerance"

    return {
        "class": class_name,
        "outer_diameter_mm": round(float(outer_diameter_mm), 2),
        "inner_diameter_mm": round(float(inner_diameter_mm), 2)
        if inner_diameter_mm is not None
        else None,
        "status": status,
        "reason": reason,
        "standard_outer_diameter_mm": standard["od"],
        "standard_inner_diameter_mm": standard["id"],
        "tolerance_mm": tolerance,
        "id_tolerance_mm": id_tolerance,
        "outer_error_mm": round(float(od_error), 2),
        "inner_error_mm": round(float(id_error), 2) if id_error is not None else None,
    }


def _draw_measurement(frame, bbox, outer_circle, inner_circle, result):
    """Draw detected circles, OD/ID values, and PASS/REJECT status on frame."""
    x1, y1, _, _ = bbox
    status = result.get("status", "REJECT")
    color = (0, 200, 0) if status == "PASS" else (0, 0, 255)

    if outer_circle is not None:
        ox, oy, outer_radius = outer_circle
        cv2.circle(frame, (x1 + ox, y1 + oy), outer_radius, (0, 255, 255), 2)
        cv2.circle(frame, (x1 + ox, y1 + oy), 2, (0, 255, 255), 3)

    if inner_circle is not None:
        ix, iy, inner_radius = inner_circle
        cv2.circle(frame, (x1 + ix, y1 + iy), inner_radius, (255, 0, 255), 2)
        cv2.circle(frame, (x1 + ix, y1 + iy), 2, (255, 0, 255), 3)

    od = result.get("outer_diameter_mm")
    bearing_id = result.get("inner_diameter_mm")
    od_text = f"OD: {od:.2f} mm" if od is not None else "OD: N/A"
    id_text = f"ID: {bearing_id:.2f} mm" if bearing_id is not None else "ID: N/A"
    status_text = f"{result.get('class', '')} {status}"

    text_x = x1
    text_y = max(25, y1 - 55)
    cv2.putText(frame, od_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, id_text, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, status_text, (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def detect_bearing_circles(frame, bbox, bearing_class, tolerance=DEFAULT_TOLERANCE_MM, draw=True):
    """
    Measure a bearing from a YOLO bounding box.

    Args:
        frame: Original BGR image frame from OpenCV.
        bbox: YOLO bbox in [x1, y1, x2, y2] format.
        bearing_class: Bearing class name, for example "608Z".
        tolerance: Measurement tolerance in millimeters.
        draw: Draw measurement visualization directly on frame when True.

    Returns:
        annotated_frame, measurement_result
    """
    class_name = _normalize_class_name(bearing_class)
    clean_bbox = _clamp_bbox(bbox, frame.shape)

    if clean_bbox is None:
        result = {
            "class": class_name,
            "outer_diameter_mm": None,
            "inner_diameter_mm": None,
            "status": "REJECT",
            "reason": "Invalid bounding box",
        }
        return frame, result

    x1, y1, x2, y2 = clean_bbox
    roi = frame[y1:y2, x1:x2].copy()

    if draw:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        _draw_text_panel(
            frame,
            [f"OpenCV ROI: {class_name}"],
            x1,
            y1 - 8,
            (255, 180, 0),
        )

    gray, blurred = _preprocess_roi(roi)

    circles = _detect_circles_hough(blurred)
    detection_method = "hough"

    if not circles:
        circles = _detect_circles_contour(gray)
        detection_method = "contour"

    outer_circle, inner_circle = _select_outer_inner_circles(circles, roi.shape)

    # Sometimes Hough finds only the OD. When the ID is missing, merge contour
    # candidates and select again so the bore still has a fallback path.
    if outer_circle is not None and inner_circle is None and detection_method == "hough":
        contour_circles = _detect_circles_contour(gray)
        if contour_circles:
            circles.extend(contour_circles)
            outer_circle, inner_circle = _select_outer_inner_circles(circles, roi.shape)
            detection_method = "hough+contour"

    if outer_circle is None:
        result = {
            "class": class_name,
            "outer_diameter_mm": None,
            "inner_diameter_mm": None,
            "status": "REJECT",
            "reason": "Circle detection failed",
            "detection_method": detection_method,
        }
        if draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            _draw_text_panel(
                frame,
                [
                    f"OpenCV: {class_name}",
                    "OD/ID circle not found",
                    "Try brighter light / closer focus",
                ],
                x1,
                y2 + 24,
                (0, 0, 255),
            )
        return frame, result

    outer_diameter_px = outer_circle[2] * 2
    pixel_to_mm = calculate_pixel_to_mm(outer_diameter_px, class_name)

    if pixel_to_mm is None:
        result = {
            "class": class_name,
            "outer_diameter_mm": None,
            "inner_diameter_mm": None,
            "status": "REJECT",
            "reason": "Missing bearing standard",
            "detection_method": detection_method,
        }
        if draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            _draw_text_panel(
                frame,
                [f"OpenCV: {class_name}", "Missing bearing standard"],
                x1,
                y2 + 24,
                (0, 0, 255),
            )
        return frame, result

    outer_diameter_mm = outer_diameter_px * pixel_to_mm
    inner_diameter_mm = None

    if inner_circle is not None:
        inner_diameter_mm = inner_circle[2] * 2 * pixel_to_mm
        inner_outer_ratio = inner_diameter_mm / outer_diameter_mm if outer_diameter_mm else 0
        if inner_outer_ratio < 0.15 or inner_outer_ratio > 0.62:
            inner_diameter_mm = None
            inner_circle = None

    result = validate_bearing_size(
        class_name,
        outer_diameter_mm,
        inner_diameter_mm,
        tolerance=tolerance,
    )
    result.update(
        {
            "outer_diameter_px": int(outer_diameter_px),
            "inner_diameter_px": int(inner_circle[2] * 2) if inner_circle is not None else None,
            "pixel_to_mm": round(float(pixel_to_mm), 6),
            "detection_method": detection_method,
        }
    )

    if draw:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0) if result["status"] == "PASS" else (0, 0, 255), 2)
        _draw_measurement(frame, clean_bbox, outer_circle, inner_circle, result)

    return frame, result
