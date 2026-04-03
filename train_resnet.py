import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import os

def train_model():
    data_dir = 'ResNet_Data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng môi trường: {device}")

    # Data Augmentation & Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Thay đổi độ sáng/tương phản để chống Overfitting
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load data
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
                   
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Hoàn tất tải dữ liệu. Tổng số Class (mệnh giá): {num_classes}")

    # Build Model from Pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    # Thay thay lớp Custom Classification cho số lượng class của chúng ta
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Optimizer & Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # AdamW giúp hội tụ nhanh
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    num_epochs = 50
    patience = 10 # Early stopping patience
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("BẮT ĐẦU HUẤN LUYỆN RESNET...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Áp dụng Scheduler và kiểm tra Early Stopping
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f">> Early stopping kích hoạt ở Epoch {epoch+1} do không cải thiện độ chính xác trên Val.")
            break

    print(f'Huấn luyện xong! Độ chính xác cao nhất (Val Acc): {best_acc:.4f}')
    
    # -----------------------------
    # Sinh biểu đồ Báo Cáo
    # -----------------------------
    print("Đang tạo các biểu đồ và báo cáo...")
    
    # 1. Đồ thị Loss / Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy History')
    plt.legend()
    plt.savefig('resnet_loss_acc_curves.png')
    plt.close()

    # Tải lại trọng số tốt nhất trước khi xuất Confusion Matrix
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_resnet.pth')
    
    # Kiểm tra metric trên tập Val
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 2. Lưu Text Report chứa F1-Score, Precision, Recall
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    with open("resnet_metrics_report.txt", "w", encoding='utf-8') as f:
        f.write("BÁO CÁO NHẬN DIỆN MỆNH GIÁ RESNET\n")
        f.write("="*40 + "\n")
        f.write(report)
        
    # 3. Vẽ Confusion Matrix (Bản đồ nhiệt nhầm lẫn)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 16)) # Để size bự do có tới 24 class
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Giá trị Thực tế (True Label)')
    plt.xlabel('Giá trị Dự đoán (Predicted Label)')
    plt.title('ResNet Confusion Matrix')
    plt.tight_layout()
    plt.savefig('resnet_confusion_matrix.png')
    plt.close()
    
    with open("resnet_classes.txt", "w") as f:
        # Ghi lại vị trí mảng class khi inference load lại
        f.write(",".join(class_names))

    print("\n[THÀNH CÔNG] Toàn bộ Model, Đồ thị và Text Metric Report đã được lưu!")

if __name__ == '__main__':
    train_model()
