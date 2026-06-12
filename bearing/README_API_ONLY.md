# Bearing API / CLI Only

Folder ini sudah diarahkan untuk fungsi saja tanpa tampilan website.

## Prediksi EfficientNet dari Terminal

```bash
python predict.py path/to/image.jpg --json
```

Input bisa berupa satu file gambar atau folder berisi gambar.

## API Upload Gambar

```bash
python predict_upload.py
```

Endpoint:

```text
POST http://127.0.0.1:8000/predict
```

Kirim gambar sebagai multipart form dengan field `image`.

Contoh:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "image=@sample.jpg"
```

## YOLO WebSocket

```bash
python yolo_ws_server.py
```

Endpoint:

```text
GET /          -> status JSON
GET /health    -> status model
WS  /ws        -> kirim frame JPEG/PNG atau JSON base64
```

Server ini tidak lagi menyediakan halaman HTML. Client eksternal bisa langsung memakai endpoint WebSocket.
