# 🏭 Bearing Inspection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ASGI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLO](https://img.shields.io/badge/YOLOv11-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com/)
[![Arduino](https://img.shields.io/badge/Arduino-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://arduino.cc/)

**Industrial-grade quality control system** for real-time bearing inspection with automated conveyor control.

---

## 📌 Overview

This project implements a complete **Computer Vision-based Quality Control System** for bearing manufacturing. The system uses YOLOv11 for real-time object detection, integrates with Arduino-controlled conveyor belt, and provides a comprehensive web dashboard for production monitoring.

| Aspect | Description |
|--------|-------------|
| **Use Case** | Manufacturing quality control, automated inspection, Industry 4.0 |
| **Key Achievement** | 99.5% detection accuracy on 1,265 image dataset |

---

## ✨ Features

| Category | Features |
|----------|----------|
| **Vision** | Real-time object detection (small/medium/large bearing) |
| **Control** | Conveyor speed control, reverse direction, auto-stop on defect |
| **Monitoring** | Production dashboard, defect rate tracking, shift management |
| **Reporting** | Inspection history, CSV export, data visualization |
| **Integration** | Arduino serial communication, USB camera support |

---

## 🖥️ Tech Stack

| Area | Technologies |
|------|--------------|
| **Backend** | FastAPI, Socket.IO ASGI, YOLOv11, OpenCV |
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js, Socket.IO |
| **Hardware** | Arduino Uno, Stepper Motor, USB Camera |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | 99.5% |
| Precision | 99.4% |
| Recall | 99.3% |
| Dataset Size | 1,265 images |
| Classes | 4 (small, medium, large, no_bearing) |

**Class Distribution:**
- small: 351 images  
- medium: 351 images  
- large: 350 images  
- no_bearing: 213 images  

---

## 🚀 Quick Start

### Prerequisites

```bash
# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Hardware Setup

### Arduino Connection

| Arduino Pin | Driver Motor |
|------------|-------------|
| D2 (DIR)   | DIR+        |
| D3 (STEP)  | PUL+        |
| GND        | DIR- / PUL- |

- Upload firmware: `sketch_arduino/sketch_arduino.ino`
- Connect USB camera to computer

---

## ▶️ Run Application

```bash
# Clone repository
git clone https://github.com/Alfpas/bearing-inspection-system.git
cd bearing-inspection-system

# Install dependencies
pip install -r requirements.txt

# Place trained model
# model/best.pt

# Run app
python app.py

# Atau langsung dengan uvicorn
uvicorn app:app --host 127.0.0.1 --port 5050
```

Open browser:
```
http://127.0.0.1:5050
```

---

## 📂 Project Structure

```
bearing-inspection-system/
│
├── app.py                    # Main application entry point
├── arduino_controller.py     # Serial communication handler
├── conveyor_controller.py    # Conveyor logic
├── camera_handler.py         # Camera & YOLO inference
├── inspection_manager.py     # Data & statistics
├── config.py                 # Configuration
│
├── model/
│   └── best.pt               # YOLO model
│
├── templates/
│   └── index.html            # Web UI
│
├── sketch_arduino/
│   └── sketch_arduino.ino    # Arduino firmware
│
├── requirements.txt
└── README.md
```

---

## 🎯 System Architecture

```
Web Browser (Dashboard)
        │
        ▼
FastAPI App (Backend + Socket.IO ASGI)
        │
   ┌────┼────┐
   ▼    ▼    ▼
Arduino YOLO   Camera
```

---

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|------------|
| /api/dashboard | GET | Production statistics |
| /api/logs | GET | Inspection history |
| /api/export | POST | Export CSV |
| /api/conveyor/start | POST | Start conveyor |
| /api/conveyor/stop | POST | Stop conveyor |
| /api/conveyor/speed | POST | Set speed |
| /api/conveyor/reverse | POST | Reverse direction |
| /detect | POST | Image detection |

---

## ⚙️ Configuration

Edit `config.py`:

```python
ARDUINO_PORT = None
BAUD_RATE = 9600
DAILY_TARGET = 1000
CONFIDENCE_THRESHOLD = 0.6

SHIFT_HOURS = {
    'Morning': (6, 14),
    'Afternoon': (14, 22),
    'Night': (22, 6)
}
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|------|---------|
| Port access denied | Close Arduino Serial Monitor |
| Camera not detected | Try index 0,1,2 |
| Model not found | Ensure `model/best.pt` exists |
| Module error | Run pip install |
| Motor not moving | Check wiring & power |

---

## 🔄 Future Improvements

- Database integration (MySQL / PostgreSQL)
- MQTT integration (IoT ready)
- Multi-camera system
- Real-time alerts
- Docker deployment

---

## 📝 License

This project is for portfolio purposes.

---

## 👨‍💻 Author

**Alfredo Pasaribu**  
Robotics Engineer | IoT | Computer Vision  

---

## ⭐ Support

If you like this project, give it a star ⭐

---

## 📧 Contact

- GitHub: https://github.com/Alfpas
- Email: alfredopasaribu110@gmail.com
- LinkedIn: www.linkedin.com/in/alfredo-pasaribu

---

<div align="center">

Built with 🐍 Python | 🔥 YOLO | ⚙️ Arduino

</div>
# bearing-inspection-system
