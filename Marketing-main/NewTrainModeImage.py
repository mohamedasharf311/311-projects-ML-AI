import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# المسارات
main_folder = "images"
model_save_path = "model.pth"
classes_save_path = "classes.json"

# إعداد التحويلات للصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def train_model():
    dataset = datasets.ImageFolder(main_folder, transform=transform)
    new_classes = sorted(dataset.classes)

    print(f"📂 عدد المنتجات الجديدة: {len(new_classes)}")
    print(f"📂 أسماء التصنيفات: {new_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # لو فيه موديل قديم
    if os.path.exists(model_save_path) and os.path.exists(classes_save_path):
        print("📦 جاري تحميل الموديل القديم...")
        
        with open(classes_save_path, "r", encoding="utf-8") as f:
            old_classes = json.load(f)

        all_classes = sorted(list(set(old_classes + new_classes)))
        print(f"📦 بعد الدمج: {len(all_classes)} تصنيف")

        checkpoint = torch.load(model_save_path, map_location=device)
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(old_classes))
        model.load_state_dict(checkpoint['model_state_dict'])

        # تعديل الطبقة النهائية لاستيعاب التصنيفات الجديدة
        model.fc = nn.Linear(num_ftrs, len(all_classes))
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

    else:
        print("🚀 بدء تدريب جديد...")
        all_classes = new_classes
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(all_classes))
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

    # تدريب
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/3 - Loss: {running_loss/len(train_loader):.4f}")

    # حفظ الموديل والتصنيفات
    torch.save({'model_state_dict': model.state_dict()}, model_save_path)
    with open(classes_save_path, "w", encoding="utf-8") as f:
        json.dump(all_classes, f, ensure_ascii=False, indent=2)

    print("✅ تم حفظ الموديل وقائمة التصنيفات بنجاح.")

if __name__ == "__main__":
    train_model()
