import os
import shutil
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(r'C:\Users\Nguye\OneDrive\Desktop\money_project')
OUT_DIR = BASE_DIR / 'yolo_merged_dataset'

VND_DIR = BASE_DIR / 'Dataset_VND' / 'VND'
INR_DIR = BASE_DIR / 'dataset_inr'
THB_DIR = BASE_DIR / 'dataset_thb'

IMG_EXTS = {'.jpg', '.jpeg', '.png'}

def convert_and_copy_label(src_label: Path, dest_label: Path):
    if not src_label.exists():
        return
    with open(src_label, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Change class ID to 0 (Banknote)
            parts[0] = '0'
            out_lines.append(' '.join(parts) + '\n')
            
    with open(dest_label, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

def process_ynd():
    splits = {'train': 'train', 'valid': 'val', 'test': 'test'}
    for vnd_split, out_split in splits.items():
        src_img_dir = VND_DIR / vnd_split / 'images'
        src_lbl_dir = VND_DIR / vnd_split / 'labels'
        
        if not src_img_dir.exists():
            continue
            
        for img_path in src_img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            
            # Copy image
            shutil.copy(img_path, OUT_DIR / 'images' / out_split / img_path.name)
            
            # Copy and convert label
            lbl_name = img_path.stem + '.txt'
            src_lbl = src_lbl_dir / lbl_name
            dest_lbl = OUT_DIR / 'labels' / out_split / lbl_name
            convert_and_copy_label(src_lbl, dest_lbl)

def process_int_thai(country_dir: Path):
    training_dir = country_dir / 'Training'
    
    if training_dir.exists():
        for class_dir in training_dir.iterdir():
            if not class_dir.is_dir(): continue
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            random.shuffle(images)
            
            # Split: 80% train, 10% val, 10% test
            n_total = len(images)
            n_val = max(1, int(n_total * 0.1))
            n_test = max(1, int(n_total * 0.1))
            if n_total < 3: n_val = 0; n_test = 0 # Corner case for very small folders
            
            test_imgs = images[:n_test]
            val_imgs = images[n_test:n_test + n_val]
            train_imgs = images[n_test + n_val:]
            
            for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
                for img_path in img_list:
                    shutil.copy(img_path, OUT_DIR / 'images' / split_name / img_path.name)
                    lbl_name = img_path.stem + '.txt'
                    src_lbl = class_dir / lbl_name
                    dest_lbl = OUT_DIR / 'labels' / split_name / lbl_name
                    convert_and_copy_label(src_lbl, dest_lbl)

def main():
    # Ensure directories exist
    for split in ['train', 'val', 'test']:
        (OUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print("Processing VND Dataset...")
    if VND_DIR.exists():
        process_ynd()
    else:
        print(f"Warning: VND_DIR {VND_DIR} not found.")

    print("Processing Indian Dataset...")
    if INR_DIR.exists(): process_int_thai(INR_DIR)
    
    print("Processing Thai Dataset...")
    if THB_DIR.exists(): process_int_thai(THB_DIR)

    # Create dataset.yaml
    yaml_content = f"""path: .
train: images/train
val: images/val
test: images/test
nc: 1
names: ['Banknote']
"""
    
    with open(OUT_DIR / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
        
    print(f"Dataset successfully created at {OUT_DIR}")

if __name__ == '__main__':
    main()
