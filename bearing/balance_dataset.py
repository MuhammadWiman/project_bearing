import os
import glob
import random
import shutil
from pathlib import Path

def get_class_files(labels_dir, target_class=0):
    files = []
    for label_file in glob.glob(str(labels_dir / "*.txt")):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            continue
            
        is_target = True
        for line in lines:
            if line.strip():
                if int(line.split()[0]) != target_class:
                    is_target = False
                    break
        if is_target:
            files.append(Path(label_file))
    return files

def balance_split(split_name, target_count):
    base_dir = Path(r"E:\Project\project_skripsi\bearing\dataset_detection_2class_finetune_pass_clean")
    labels_dir = base_dir / "labels" / split_name
    images_dir = base_dir / "images" / split_name
    train_labels_dir = base_dir / "labels" / "train"
    train_images_dir = base_dir / "images" / "train"
    
    reject_files = get_class_files(labels_dir, target_class=0)
    print(f"[{split_name}] Ditemukan {len(reject_files)} file gambar yang murni 'Reject'.")
    
    if len(reject_files) > target_count:
        excess_count = len(reject_files) - target_count
        # Memilih secara acak file Reject berlebih untuk dipindah
        files_to_move = random.sample(reject_files, excess_count)
        
        moved = 0
        for label_path in files_to_move:
            base_name = label_path.stem
            
            # Cari file gambar yang sesuai
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.webp', '.bmp']:
                temp = images_dir / f"{base_name}{ext}"
                if temp.exists():
                    img_path = temp
                    break
                    
            if img_path:
                # Pindahkan label
                shutil.move(str(label_path), str(train_labels_dir / label_path.name))
                # Pindahkan image
                shutil.move(str(img_path), str(train_images_dir / img_path.name))
                moved += 1
                
        print(f"[{split_name}] Berhasil memindahkan {moved} file 'Reject' berlebih ke folder Train.")
    else:
        print(f"[{split_name}] Jumlah 'Reject' sudah <= {target_count}, tidak ada yang dipindah.")

def main():
    random.seed(42)  # Agar acakannya konsisten
    print("Memulai proses penyeimbangan Downsampling pada Valid dan Test...")
    # Target: Samakan Reject dengan jumlah Pass
    balance_split("valid", 61)
    balance_split("test", 36)
    print("Proses selesai!")

if __name__ == '__main__':
    main()
