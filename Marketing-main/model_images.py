import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.serialization

# الوثوق في LabelEncoder عند تحميل النموذج
torch.serialization.add_safe_globals([LabelEncoder])

# المسارات
images_path = r"D:\MyDream\archive\fashion-dataset\images"
csv_path = r"D:\MyDream\archive\fashion-dataset\styles.csv"
model_save_path = "D:/MyDream/multilabel_model.pth"

# الأعمدة المستهدفة
targets = ['gender', 'subCategory', 'articleType', 'baseColour']

# تحميل CSV
df = pd.read_csv(csv_path, on_bad_lines='skip')

# حذف الصفوف اللي مافيهاش صورة
df.dropna(subset=['id'], inplace=True)
df['image_path'] = df['id'].astype(str).apply(lambda x: os.path.join(images_path, x + ".jpg"))
df = df[df['image_path'].apply(os.path.exists)]

# تحميل النموذج القديم لو موجود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(model_save_path):
    try:
        checkpoint = torch.load(model_save_path, map_location=device)
        encoders = checkpoint['label_encoders']
        num_classes_dict = checkpoint['num_classes_dict']
        print(" Existing model found – filtering data to match it...")

        for col in targets:
            allowed_classes = set(encoders[col].classes_)
            df = df[df[col].isin(allowed_classes)]
            df[col + '_label'] = encoders[col].transform(df[col])
    except Exception as e:
        print("Failed to load previous model. Starting fresh. Reason:", e)
        checkpoint = None
        encoders = {}
        num_classes_dict = {}
else:
    checkpoint = None
    print(" No saved model found – starting new training...")

# تدريب جديد لو مفيش إنكودرات
if not encoders:
    encoders = {}
    for col in targets:
        enc = LabelEncoder()
        df[col + '_label'] = enc.fit_transform(df[col])
        encoders[col] = enc
    num_classes_dict = {
        key: df[key + '_label'].nunique()
        for key in targets
    }

# Dataset
class FashionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = {target: torch.tensor(row[target + "_label"], dtype=torch.long) for target in self.targets}
        return image, labels

# الموديل
class MultiHeadCNN(nn.Module):
    def __init__(self, num_classes_dict):
        super(MultiHeadCNN, self).__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()
        self.backbone = base

        self.heads = nn.ModuleDict()
        for key, num_classes in num_classes_dict.items():
            self.heads[key] = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        features = self.backbone(x)
        outputs = {key: head(features) for key, head in self.heads.items()}
        return outputs

# التحويلات
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# تقسيم البيانات
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = FashionDataset(train_df, transform=transform)
val_dataset = FashionDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# تحميل النموذج
model = MultiHeadCNN(num_classes_dict).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# تحميل weights لو متاحة
if checkpoint:
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(" Model and optimizer loaded.")
    except Exception as e:
        print(" Failed to load weights. Reason:", e)

# التدريب
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        optimizer.zero_grad()
        outputs = model(images)

        loss = sum(criterion(outputs[k], labels[k]) for k in outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

# حفظ النموذج
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'label_encoders': encoders,
    'num_classes_dict': num_classes_dict
}, model_save_path)
print(" Model saved successfully.")
