import os
import glob
import random
from pathlib import Path

def main():
    random.seed(42)
    base_dir = Path(r"E:\Project\project_skripsi\bearing\dataset_detection_2class_finetune_pass_clean")
    labels_dir = base_dir / "labels" / "train"
    images_dir = base_dir / "images" / "train"
    
    # 1. Hapus augmentasi Pass (_aug)
    print("Menghapus gambar augmentasi (_aug) di folder Train...")
    aug_labels = glob.glob(str(labels_dir / "*_aug*.txt"))
    for label_path in aug_labels:
        base_name = Path(label_path).stem
        os.remove(label_path)
        # Cari dan hapus image
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.webp', '.bmp']:
            img_path = images_dir / f"{base_name}{ext}"
            if img_path.exists():
                os.remove(img_path)
                break
                
    print(f"Berhasil menghapus {len(aug_labels)} gambar augmentasi.")
    
    # 2. Kurangi Reject di Train menjadi tepat 433
    reject_files = []
    for label_file in glob.glob(str(labels_dir / "*.txt")):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        is_reject = True
        for line in lines:
            if line.strip():
                if int(line.split()[0]) != 0:
                    is_reject = False
                    break
        if is_reject:
            reject_files.append(Path(label_file))
            
    print(f"Ditemukan {len(reject_files)} gambar murni Reject di folder Train.")
    
    target_reject = 433
    if len(reject_files) > target_reject:
        excess = len(reject_files) - target_reject
        files_to_delete = random.sample(reject_files, excess)
        
        deleted = 0
        for label_path in files_to_delete:
            base_name = label_path.stem
            os.remove(label_path)
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.webp', '.bmp']:
                img_path = images_dir / f"{base_name}{ext}"
                if img_path.exists():
                    os.remove(img_path)
                    break
            deleted += 1
            
        print(f"Berhasil menghapus {deleted} gambar Reject.")
        print(f"Sisa Reject di folder Train sekarang adalah: {target_reject}")
    else:
        print(f"Jumlah Reject sudah <= {target_reject}, tidak ada yang dihapus.")

if __name__ == '__main__':
    main()
