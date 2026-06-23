# config.py
import os

# Model
MODEL_PATH = 'model/best.pt'
MODEL_1_PATH = '../bearing/runs/detect/bearing_yolov8_v5/weights/best.pt'
MODEL_1_CONFIDENCE = 0.2
DETECTION_CONFIDENCE = 0.4
DETECTION_IMAGE_SIZE = 640
MODEL_1_IMAGE_SIZE = 224
MODEL_2_DEFAULT_BEARING_TYPE = '608Z'

# Model kelengkapan di folder bearing memakai class Pass dan NOT_GOOD.
# Jika class model berubah menjadi part-part spesifik, isi daftar ini dengan
# class wajib yang harus muncul agar status menjadi "Lengkap".
REQUIRED_ASSEMBLY_CLASSES = []

# Arduino
ARDUINO_PORT = None  # Ubah ke None agar sistem melakukan scanning otomatis (auto-discovery)
BAUD_RATE = 9600

# Production target
DAILY_TARGET = 1000
INSPECTION_DB_PATH = 'inspection_qc.db'

# Live inspection refresh interval in seconds.
# Naikkan nilainya jika tampilan live terlalu cepat berubah.
LIVE_INSPECTION_INTERVAL = 0.1

# Jumlah frame yang dipakai untuk voting status live agar hasil tidak mudah
# berubah karena satu frame blur / noise.
STABILITY_WINDOW = 3
STABILITY_LOCK_COUNT = 2
ASSEMBLY_STABILITY_FRAMES = STABILITY_WINDOW
DETECTION_STABILITY_FRAMES = STABILITY_WINDOW
DETECTION_CLASS_STABILITY_FRAMES = STABILITY_WINDOW

# ID/OpenCV inner-circle mudah berubah karena pantulan dan bayangan. Biarkan
# False agar PASS/REJECT ditentukan oleh OD dan model kelengkapan saja.
USE_ID_FOR_VALIDATION = False

# Area bbox membantu saat kelas YOLO kadang tertukar.
# Saat 2 bearing luasnya serupa, class bisa disamakan berdasarkan hasil model.
# Batas Big dipakai sebagai koreksi ukuran agar bbox besar dihitung 6301Z.
# Saat 3 bearing terdeteksi: kecil 688Z, tengah 608Z, besar 6301Z.
USE_BBOX_SIZE_CLASS_CORRECTION = True
USE_TWO_BEARING_SLOT_CLASS_CORRECTION = True
SIMILAR_BBOX_AREA_RATIO = 0.25
BIG_BEARING_MIN_BBOX_WIDTH = 145

# Untuk inspeksi 3 bearing, kelengkapan dianggap PASS jika 3 ukuran utama
# terdeteksi. Ini menghindari model kelengkapan menolak part lengkap karena
# satu bbox besar sesekali terbaca NOT_GOOD.
ASSEMBLY_PASS_BY_DETECTED_BEARINGS = True
REQUIRED_BEARING_COUNT = 3
MAX_BEARING_DETECTIONS = 3
DETECTION_DEDUP_IOU_THRESHOLD = 0.25

# Model 1 tetap menjadi pembaca utama kelengkapan. Fallback 3 bearing hanya
# dipakai saat Model 1 tidak membaca objek sama sekali.
USE_MODEL1_FOR_ASSEMBLY_DECISION = True
SHOW_MODEL1_BOX_WHEN_BEARINGS_DECIDE = True
USE_BEARING_FALLBACK_WHEN_MODEL1_EMPTY = True
USE_BEARING_FALLBACK_FOR_MODEL1_MISSING = True

# Colors
COLORS = {
    'no_bearing': (128, 128, 128),
    'small': (0, 255, 0),
    'small_bearing': (0, 255, 0),
    'SMALL_BEARING': (0, 255, 0),
    'medium': (0, 165, 255),
    'medium_bearing': (0, 165, 255),
    'MEDIUM_BEARING': (0, 165, 255),
    'large': (0, 0, 255),
    'large_bearing': (0, 0, 255),
    'LARGE_BEARING': (0, 0, 255),
    'big_bearings': (0, 0, 255),
    'BIG_BEARINGS': (0, 0, 255)
}

CLASS_STATUS = {
    'small': 'OK',
    'small_bearing': 'OK',
    'SMALL_BEARING': 'OK',
    'medium': 'WARNING',
    'medium_bearing': 'WARNING',
    'MEDIUM_BEARING': 'WARNING',
    'large': 'REJECT',
    'large_bearing': 'REJECT',
    'LARGE_BEARING': 'REJECT',
    'big_bearings': 'REJECT',
    'BIG_BEARINGS': 'REJECT',
    'no_bearing': 'NO_PRODUCT'
}

# Shift hours
SHIFT_HOURS = {
    'Morning': (6, 14),
    'Afternoon': (14, 22),
    'Night': (22, 6)
}

# Gemini Vision API
GEMINI_API_KEY = ""
USE_GEMINI_VERIFICATION = True
