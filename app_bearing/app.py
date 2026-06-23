# app.py - Main FastAPI App
import asyncio
import csv
import io
import os
from pathlib import Path
from datetime import datetime

import socketio
from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "Ultralytics"))

from config import ARDUINO_PORT, BAUD_RATE, MODEL_PATH, MODEL_1_PATH, INSPECTION_DB_PATH
from arduino_controller import ArduinoController
from conveyor_controller import ConveyorController
from inspection_manager import InspectionManager
from camera_handler import CameraHandler
from gemini_verifier import GeminiVerifier


fastapi_app = FastAPI(title="Industrial Bearing Inspection System")
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Initialize components
arduino = ArduinoController(ARDUINO_PORT, BAUD_RATE)
conveyor = ConveyorController(arduino)
inspection = InspectionManager(
    str(BASE_DIR / INSPECTION_DB_PATH),
    legacy_json_file=str(BASE_DIR / "inspection_log.json"),
)
gemini = GeminiVerifier()

async def verify_defect_async(record_id, img_bytes, status, class_name):
    # Dijalankan di background agar tidak memblokir respon UI
    reason = await asyncio.to_thread(gemini.verify_defect, img_bytes, status, class_name)
    inspection.update_ai_reason(record_id, reason)



def resolve_app_path(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str((BASE_DIR / path).resolve())


camera = CameraHandler(
    resolve_app_path(MODEL_PATH),
    conveyor,
    inspection,
    assembly_model_path=resolve_app_path(MODEL_1_PATH),
)
socket_loop = None


@fastapi_app.on_event("startup")
async def startup():
    global socket_loop
    socket_loop = asyncio.get_running_loop()
    arduino.connect()


@fastapi_app.on_event("shutdown")
async def shutdown():
    camera.stop()
    arduino.close()


def emit_from_thread(event, data):
    if socket_loop and socket_loop.is_running():
        asyncio.run_coroutine_threadsafe(sio.emit(event, data), socket_loop)


def send_frame(frame_base64, detections, total, opencv_frame_base64=None, assembly=None):
    emit_from_thread(
        "frame",
        {
            "image": frame_base64,
            "yolo_image": frame_base64,
            "opencv_image": opencv_frame_base64 or frame_base64,
            "detections": detections,
            "total": total,
            "assembly": assembly,
        },
    )


# Socket.IO handlers
@sio.event
async def connect(sid, environ):
    print("Client connected")


@sio.on("start_camera")
async def handle_start_camera(sid, data):
    data = data or {}
    index = data.get("index", 1)
    result = camera.start(index, callback=send_frame)
    await sio.emit("camera_status", result, to=sid)


@sio.on("stop_camera")
async def handle_stop_camera(sid):
    result = camera.stop()
    await sio.emit("camera_status", result, to=sid)


@sio.on("test_cameras")
async def handle_test_cameras(sid):
    import cv2

    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append({"index": i, "name": f"Camera {i}"})
            cap.release()
    await sio.emit("cameras_available", {"cameras": available}, to=sid)


@sio.on("camera_settings")
async def handle_camera_settings(sid, data):
    settings = camera.update_settings(data or {})
    await sio.emit("camera_settings", settings, to=sid)


# Routes
@fastapi_app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@fastapi_app.get("/api/dashboard")
async def dashboard(scope: str = "today"):
    return inspection.get_statistics(scope)


@fastapi_app.get("/api/model-info")
async def model_info():
    assembly_model = camera.assembly_model
    return {
        "model_1_path": getattr(assembly_model, "model_path", None),
        "model_1_loaded": bool(getattr(assembly_model, "model", None)),
        "model_1_names": getattr(assembly_model, "names", {}),
        "model_2_path": resolve_app_path(MODEL_PATH),
    }


@fastapi_app.get("/api/camera/settings")
async def camera_settings():
    return camera.get_settings()


@fastapi_app.post("/api/camera/settings")
async def update_camera_settings(payload: dict = Body(default=None)):
    return camera.update_settings(payload or {})


@fastapi_app.get("/api/logs")
async def get_logs(
    class_filter: str | None = Query(default=None, alias="class"),
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
):
    logs = inspection.get_logs(class_filter, date_from, date_to, limit)
    return {"logs": logs, "total": len(logs)}


@fastapi_app.get("/api/reports")
async def get_reports(date_from: str | None = None, date_to: str | None = None):
    return inspection.get_report(date_from, date_to)


@fastapi_app.post("/api/export")
async def export():
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["timestamp", "class", "confidence", "status", "shift", "image_name"],
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(inspection.logs)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@fastapi_app.post("/detect")
async def detect(image: UploadFile | None = File(default=None)):
    if not image:
        raise HTTPException(status_code=400, detail="No image")

    img_bytes = await image.read()
    detections, img_base64, assembly = camera.detect_image(img_bytes)
    for detection in detections:
        record = inspection.add(
            detection["class"],
            detection["confidence"],
            image.filename,
            measurement=detection.get("measurement"),
            m1_confidence=assembly.get("assembly_confidence"),
        )
        status = detection.get("status") or (detection.get("measurement") or {}).get("status")
        if status == "REJECT":
            asyncio.create_task(verify_defect_async(record["id"], img_bytes, status, detection["class"]))
            
    return {
        "success": True,
        "detections": detections,
        "total": len(detections),
        "image": img_base64,
        "assembly": assembly,
    }


# Conveyor routes
@fastapi_app.get("/api/conveyor/status")
async def conveyor_status():
    return conveyor.get_status()


@fastapi_app.post("/api/conveyor/start")
async def conveyor_start():
    result = conveyor.start()
    await sio.emit("conveyor_event", {"action": "started", "message": "Conveyor started"})
    return result


@fastapi_app.post("/api/conveyor/stop")
async def conveyor_stop(payload: dict | None = Body(default=None)):
    reason = "Manual stop"
    if payload:
        reason = payload.get("reason", reason)
    result = conveyor.stop(reason)
    await sio.emit(
        "conveyor_event",
        {"action": "stopped", "message": f"Conveyor stopped: {reason}"},
    )
    return result


@fastapi_app.post("/api/conveyor/speed")
async def conveyor_speed(payload: dict | None = Body(default=None)):
    speed = 70
    if payload:
        speed = payload.get("speed", speed)
    result = conveyor.set_speed(speed)
    await sio.emit(
        "conveyor_event",
        {"action": "speed_changed", "message": f"Speed: {speed}%"},
    )
    return result


@fastapi_app.post("/api/conveyor/auto-stop")
async def conveyor_auto_stop(payload: dict | None = Body(default=None)):
    enabled = True
    if payload:
        enabled = payload.get("enabled", enabled)
    return conveyor.set_auto_stop(enabled)


@fastapi_app.post("/api/conveyor/reset-counter")
async def conveyor_reset():
    result = conveyor.reset_counter()
    await sio.emit("conveyor_event", {"action": "counter_reset", "message": "Counter reset"})
    return result


@fastapi_app.post("/api/conveyor/defect")
async def conveyor_defect():
    result = conveyor.add_defect()
    await sio.emit("defect_alert", {"message": "DEFECT DETECTED!"})
    return result


@fastapi_app.post("/api/conveyor/reverse")
async def conveyor_reverse():
    result = conveyor.reverse()
    await sio.emit(
        "conveyor_event",
        {"action": "direction_changed", "message": "Direction reversed"},
    )
    return result


@fastapi_app.get("/api/ports")
async def get_ports():
    import serial.tools.list_ports

    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append({"device": port.device, "description": port.description})
    return {"ports": ports, "arduino_connected": arduino.connected}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5050)
