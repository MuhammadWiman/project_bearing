import os
import glob
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

def augment_image(image, aug_type):
    if aug_type == 1:
        # Brightness and Contrast
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))
    elif aug_type == 2:
        # Color and Sharpness
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(random.uniform(0.5, 2.0))
        # Add slight blur sometimes
        if random.random() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return image

def main():
    base_dir = Path(r"E:\Project\project_skripsi\bearing\dataset_detection_2class_finetune_pass_clean")
    labels_dir = base_dir / "labels" / "train"
    images_dir = base_dir / "images" / "train"

    label_files = glob.glob(str(labels_dir / "*.txt"))
    pass_only_files = []

    # Find images that ONLY contain "Pass" (class 1)
    for label_path in label_files:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            continue
            
        is_pass_only = True
        for line in lines:
            if line.strip():
                class_id = int(line.split()[0])
                if class_id != 1:
                    is_pass_only = False
                    break
                    
        if is_pass_only:
            pass_only_files.append(Path(label_path))

    print(f"Ditemukan {len(pass_only_files)} gambar yang hanya berisi kelas 'Pass'.")
    print("Memulai proses augmentasi (membuat 2 variasi baru per gambar)...")

    count_augmented = 0
    for label_path in pass_only_files:
        # Find corresponding image
        base_name = label_path.stem
        
        # Try to find the image with common extensions
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            temp_path = images_dir / f"{base_name}{ext}"
            if temp_path.exists():
                img_path = temp_path
                break
                
        if not img_path:
            continue

        try:
            original_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Gagal membuka {img_path}: {e}")
            continue

        # Create 2 augmentations
        for i in range(1, 3):
            aug_img = augment_image(original_img.copy(), aug_type=i)
            new_base_name = f"{base_name}_aug{i}"
            
            new_img_path = images_dir / f"{new_base_name}{img_path.suffix}"
            new_label_path = labels_dir / f"{new_base_name}.txt"

            # Save augmented image
            aug_img.save(new_img_path)
            
            # Copy label (geometry doesn't change because we only do pixel-level augmentation)
            shutil.copy(label_path, new_label_path)
            count_augmented += 1

    print(f"Selesai! Berhasil membuat {count_augmented} gambar augmentasi baru.")

if __name__ == '__main__':
    main()
