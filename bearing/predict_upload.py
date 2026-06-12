import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import cgi

from PIL import Image
import torch

from predict import build_transform, load_model


def predict_uploaded_image(model, image_bytes, transform, classes, threshold, device):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu()

    probabilities = {
        label: round(float(prob), 6)
        for label, prob in zip(classes, probs)
    }
    predicted_labels = [
        label
        for label, prob in probabilities.items()
        if prob >= threshold
    ]

    return predicted_labels, probabilities



class UploadHandler(BaseHTTPRequestHandler):
    model = None
    transform = None
    classes = None
    threshold = 0.5
    device = None

    def do_GET(self):
        self.respond_json({
            "service": "bearing-upload-predict-api",
            "status": "running",
            "message": "API-only mode. Send a multipart POST request to /predict with field name image.",
            "predict": "/predict",
        })

    def do_POST(self):
        try:
            if self.path != "/predict":
                self.respond_json({"error": "Endpoint tidak ditemukan. Gunakan /predict."}, status=404)
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                    "CONTENT_LENGTH": self.headers.get("Content-Length"),
                },
            )
            field = form["image"] if "image" in form else None

            if field is None or not field.file:
                self.respond_json({"error": "File gambar belum dipilih."}, status=400)
            else:
                image_bytes = field.file.read()
                predicted_labels, probabilities = predict_uploaded_image(
                    self.model,
                    image_bytes,
                    self.transform,
                    self.classes,
                    self.threshold,
                    self.device,
                )
                self.respond_json({
                    "success": True,
                    "filename": field.filename or "uploaded-image",
                    "threshold": self.threshold,
                    "predicted_labels": predicted_labels,
                    "probabilities": probabilities,
                })
        except Exception as exc:
            self.respond_json({"error": str(exc)}, status=500)

    def respond_json(self, payload, status=200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args():
    parser = argparse.ArgumentParser(description="Jalankan API prediksi foto bearing tanpa tampilan website.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pth"),
        help="Path checkpoint model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold multi-label prediction.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device inference.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host API server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port API server.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.checkpoint}")

    device = torch.device(args.device)
    UploadHandler.model, UploadHandler.classes = load_model(args.checkpoint, device)
    UploadHandler.transform = build_transform()
    UploadHandler.threshold = args.threshold
    UploadHandler.device = device

    server = ThreadingHTTPServer((args.host, args.port), UploadHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Server API jalan di {url}")
    print(f"Endpoint prediksi: {url}/predict")
    print("Tekan Ctrl+C untuk berhenti.")
    server.serve_forever()


if __name__ == "__main__":
    main()
