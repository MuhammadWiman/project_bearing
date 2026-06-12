import argparse
import base64
import hashlib
import json
import os
import struct
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

from ultralytics import YOLO


DEFAULT_MODEL = PROJECT_ROOT / "runs/detect/bearing_yolov8/weights/best.pt"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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


def decode_image(message):
    if isinstance(message, str):
        if not message.strip():
            raise ValueError("Payload WebSocket kosong.")

        payload = json.loads(message)
        image_data = payload["image"]
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = message

    if not image_bytes:
        raise ValueError("Frame gambar kosong. Kirim binary JPEG/PNG atau JSON base64 yang valid.")

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    if array.size == 0:
        raise ValueError("Frame gambar kosong. Buffer image 0 byte.")

    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Frame gambar tidak bisa dibaca.")
    return frame


def encode_annotated_image(result):
    annotated = result.plot()
    ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise ValueError("Gagal encode hasil anotasi.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def run_inference(server, frame):
    started = time.perf_counter()
    results = server.model.predict(
        source=frame,
        imgsz=server.imgsz,
        conf=server.conf,
        iou=server.iou,
        device=server.device,
        verbose=False,
    )
    result = results[0]
    return {
        "type": "prediction",
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "detections": [box_to_dict(box, server.model.names) for box in result.boxes],
        "annotated_image": encode_annotated_image(result),
    }


class WebSocketConnection:
    def __init__(self, request):
        self.request = request

    def recv(self):
        header = self.request.rfile.read(2)
        if len(header) < 2:
            return None

        first_byte, second_byte = header
        opcode = first_byte & 0x0F
        masked = bool(second_byte & 0x80)
        length = second_byte & 0x7F

        if length == 126:
            length = struct.unpack("!H", self.request.rfile.read(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self.request.rfile.read(8))[0]

        mask = self.request.rfile.read(4) if masked else b""
        payload = self.request.rfile.read(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))

        if opcode == 0x8:
            return None
        if opcode == 0x1:
            return payload.decode("utf-8")
        if opcode == 0x2:
            return payload
        if opcode == 0x9:
            self.send(payload, opcode=0xA)
            return self.recv()
        return b""

    def send_json(self, payload):
        self.send(json.dumps(payload, separators=(",", ":")).encode("utf-8"), opcode=0x1)

    def send(self, payload, opcode=0x1):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        length = len(payload)
        header = bytearray([0x80 | opcode])
        if length < 126:
            header.append(length)
        elif length < 65536:
            header.extend([126])
            header.extend(struct.pack("!H", length))
        else:
            header.extend([127])
            header.extend(struct.pack("!Q", length))

        self.request.wfile.write(header + payload)
        self.request.wfile.flush()


class BearingYoloServer(ThreadingHTTPServer):
    daemon_threads = True
    model = None
    imgsz = 224
    conf = 0.25
    iou = 0.45
    device = None


class BearingYoloHandler(BaseHTTPRequestHandler):
    server_version = "BearingYoloWebSocket/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_json({
                "service": "bearing-yolo-websocket-api",
                "status": "running",
                "message": "API-only mode. No website UI is served.",
                "health": "/health",
                "websocket": "/ws",
            })
            return
        if parsed.path == "/health":
            self.send_json({"status": "ok", "model": str(self.server.model_path)})
            return
        if parsed.path == "/ws":
            self.handle_websocket()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_json(self, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def handle_websocket(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing Sec-WebSocket-Key")
            return

        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode("ascii")).digest())
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept.decode("ascii"))
        self.end_headers()

        ws = WebSocketConnection(self)
        while True:
            message = ws.recv()
            if message is None:
                break

            try:
                frame = decode_image(message)
                ws.send_json(run_inference(self.server, frame))
            except Exception as exc:
                ws.send_json({"type": "error", "message": str(exc)})

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jalankan YOLO bearing model dan stream hasil inference ke client via WebSocket."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path weight YOLO .pt hasil training.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host server.")
    parser.add_argument("--port", type=int, default=8765, help="Port server.")
    parser.add_argument("--imgsz", type=int, default=224, help="Ukuran input YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold NMS.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device inference. Contoh: cpu atau 0. Default auto dari Ultralytics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model YOLO tidak ditemukan: {model_path}")

    server = BearingYoloServer((args.host, args.port), BearingYoloHandler)
    server.model_path = model_path
    server.model = YOLO(str(model_path))
    server.imgsz = args.imgsz
    server.conf = args.conf
    server.iou = args.iou
    server.device = args.device

    print(f"Model: {model_path}")
    print(f"Server: http://{args.host}:{args.port}")
    print("WebSocket: /ws")
    print("Tekan Ctrl+C untuk berhenti.")
    server.serve_forever()


if __name__ == "__main__":
    main()
