from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    print("Memuat model v5...")
    model = YOLO('runs/detect/bearing_yolov8_v5/weights/best.pt')

    print("Menjalankan validasi untuk mengambil data Confusion Matrix...")
    # Menjalankan validasi tanpa plot bawaan agar lebih cepat, meskipun bawaan tetap tersimpan
    metrics = model.val(data='dataset_detection_2class_finetune_pass_clean/data.yaml')

    # Mengambil matriks asli dari Ultralytics (biasanya bentuknya 3x3 karena ada background di akhir)
    cm_raw = metrics.confusion_matrix.matrix
    
    # Kita potong hanya mengambil 2x2 (Baris 0-1, Kolom 0-1) yang berisi Reject dan Pass
    cm_2x2 = cm_raw[:2, :2]
    
    class_names = ['Reject', 'Pass']

    print("Menggambar matriks...")
    
    # Plot 1: Raw Counts (Angka Asli)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_2x2, annot=True, fmt='.0f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    
    out_path_raw = 'runs/detect/bearing_yolov8_v5/custom_confusion_matrix.png'
    plt.savefig(out_path_raw, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Normalized (Persentase)
    # Hati-hati pembagian dengan nol jika tidak ada data
    row_sums = cm_2x2.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm_2x2.astype('float'), row_sums, out=np.zeros_like(cm_2x2, dtype=float), where=row_sums!=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Normalized Confusion Matrix', fontsize=16)
    
    out_path_norm = 'runs/detect/bearing_yolov8_v5/custom_confusion_matrix_normalized.png'
    plt.savefig(out_path_norm, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nBerhasil! File telah disimpan di:")
    print(f"- {out_path_raw}")
    print(f"- {out_path_norm}")

if __name__ == '__main__':
    main()
