import os
import io
from PIL import Image
import google.generativeai as genai
from config import GEMINI_API_KEY, USE_GEMINI_VERIFICATION

class GeminiVerifier:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.enabled = USE_GEMINI_VERIFICATION and bool(self.api_key)
        
        if self.enabled:
            genai.configure(api_key=self.api_key)
            # Menggunakan model Flash yang ringan dan sangat cepat untuk penglihatan gambar (Vision)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def verify_defect(self, image_bytes, yolo_status="REJECT", bearing_type="Unknown"):
        if not self.enabled or not self.model:
            return "Gemini API Key belum dimasukkan di config.py"
            
        try:
            # Mengonversi gambar OpenCV (bytes) ke format bantal (PIL) agar bisa dibaca Gemini
            img = Image.open(io.BytesIO(image_bytes))
            
            prompt = (
                f"Kamu adalah AI Quality Control (QC) Inspector pabrik bearing. "
                f"Sistem kamera awal mendeteksi tipe bearing '{bearing_type}' ini berstatus: {yolo_status}. "
                "Tolong periksa gambar bearing ini dengan sangat teliti. "
                "Apakah kamu setuju dengan status tersebut? Coba perhatikan apakah ada cacat fisik seperti karet/seal penyok, terlepas, berkarat, atau kotoran ekstrem. "
                "Berikan analisis singkat (maksimal 2 kalimat) untuk memverifikasi kondisi aktualnya dengan bahasa Indonesia yang formal."
            )
            
            response = self.model.generate_content([prompt, img])
            return response.text.strip()
            
        except Exception as e:
            return f"Sistem AI gagal memverifikasi: {str(e)}"
