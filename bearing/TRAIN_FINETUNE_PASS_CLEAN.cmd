cd /d E:\Project\project_skripsi\bearing
python train_yolov8.py --data dataset_detection_2class_finetune_pass_clean/data.yaml --model runs/detect/bearing_yolov8/weights/best.pt --epochs 30 --imgsz 224 --batch 16 --workers 0 --patience 10 --project runs/detect --name bearing_yolov8_finetune_pass_clean --validate
