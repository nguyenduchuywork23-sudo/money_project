import os
from pathlib import Path
from PIL import Image
import random

# Tăng giới hạn kích thước hình ảnh nếu cần tránh Image Decompression Bomb ERROR
Image.MAX_IMAGE_PIXELS = None 

BASE_DIR = Path(r'C:\Users\Nguye\OneDrive\Desktop\money_project')
OUT_DIR = BASE_DIR / 'ResNet_Data'

VND_DIR = BASE_DIR / 'Dataset_VND' / 'VND'
INR_DIR = BASE_DIR / 'dataset_inr'
THB_DIR = BASE_DIR / 'dataset_thb'

IMG_EXTS = {'.jpg', '.jpeg', '.png'}

# Danh sách classes của VND theo data.yaml (từ id 0 tới 8)
VND_CLASSES = [
    'VND_1000', 'VND_10000', 'VND_100000', 
    'VND_2000', 'VND_20000', 'VND_200000', 
    'VND_5000', 'VND_50000', 'VND_500000'
]

def crop_and_save(img_path: Path, src_lbl: Path, split: str, class_name: str, instance_id: str):
    if not src_lbl.exists(): return
    try:
        img = Image.open(img_path)
        img = img.convert("RGB") # Ensure no transparency issues
        W, H = img.size
    except Exception as e:
        print(f"Lỗi đọc ảnh {img_path}: {e}")
        return
        
    with open(src_lbl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    dest_dir = OUT_DIR / split / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 5:
            # Format: class_id x_center y_center width height
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            x1 = max(0, int((x_center - w/2) * W))
            y1 = max(0, int((y_center - h/2) * H))
            x2 = min(W, int((x_center + w/2) * W))
            y2 = min(H, int((y_center + h/2) * H))
            
            if x2 > x1 and y2 > y1:
                cropped = img.crop((x1, y1, x2, y2))
                dest_path = dest_dir / f"{instance_id}_crop{i}.jpg"
                cropped.save(dest_path)

def process_ynd():
    splits = {'train': 'train', 'valid': 'val', 'test': 'val'} # Gộp test vào val cho mô hình classification ImageFolder
    for vnd_split, out_split in splits.items():
        src_img_dir = VND_DIR / vnd_split / 'images'
        src_lbl_dir = VND_DIR / vnd_split / 'labels'
        
        if not src_img_dir.exists(): continue
            
        for img_path in src_img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS: continue
            
            lbl_name = img_path.stem + '.txt'
            src_lbl = src_lbl_dir / lbl_name
            if not src_lbl.exists(): continue
            
            with open(src_lbl, 'r', encoding='utf-8') as f: 
                lines = f.readlines()
            
            try:
                img = Image.open(img_path)
                img = img.convert("RGB")
                W, H = img.size
            except: 
                continue
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        # Lấy class_name tương ứng cho phân loại ResNet
                        if cls_id < len(VND_CLASSES):
                            class_name = VND_CLASSES[cls_id]
                            x_center, y_center, w, h = map(float, parts[1:5])
                            x1 = max(0, int((x_center - w/2) * W))
                            y1 = max(0, int((y_center - h/2) * H))
                            x2 = min(W, int((x_center + w/2) * W))
                            y2 = min(H, int((y_center + h/2) * H))
                            
                            if x2 > x1 and y2 > y1:
                                dest_dir = OUT_DIR / out_split / class_name
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                cropped = img.crop((x1, y1, x2, y2))
                                dest_path = dest_dir / f"{img_path.stem}_crop{i}.jpg"
                                cropped.save(dest_path)
                    except:
                        pass

def process_int_thai(country_dir: Path):
    training_dir = country_dir / 'Training'
    if training_dir.exists():
        for class_dir in training_dir.iterdir():
            if not class_dir.is_dir(): continue
            class_name = class_dir.name
            images = [img for img in class_dir.iterdir() if img.suffix.lower() in IMG_EXTS]
            random.shuffle(images)
            
            num_val = max(1, int(len(images) * 0.2)) # at least 1 image for val, mostly 20%
            if len(images) < 2: num_val = 0
            val_imgs = images[:num_val]
            train_imgs = images[num_val:]
            
            for split_name, img_list in [('val', val_imgs), ('train', train_imgs)]:
                for img_path in img_list:
                    src_lbl = class_dir / (img_path.stem + '.txt')
                    crop_and_save(img_path, src_lbl, split_name, class_name, img_path.stem)

def main():
    print("Khởi tạo cấu trúc cắt ảnh Tầng 2...")
    for split in ['train', 'val']:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    print("Đang crop ảnh Dataset VND...")
    if VND_DIR.exists():
        process_ynd()
    else:
        print(f"Cảnh báo: Không tìm thấy thư mục {VND_DIR}")

    print("Đang crop ảnh Dataset Indian...")
    if INR_DIR.exists(): process_int_thai(INR_DIR)
    
    print("Đang crop ảnh Dataset Thai...")
    if THB_DIR.exists(): process_int_thai(THB_DIR)

    print(f"\nHoàn tất! ResNet_Data được tạo tại: {OUT_DIR}")

if __name__ == '__main__':
    main()
