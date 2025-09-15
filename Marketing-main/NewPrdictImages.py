import torch
from torchvision import transforms, models
from PIL import Image
import json
import os

# المسارات
MODEL_PATH = "model.pth"
CLASSES_PATH = "classes.json"

# تحميل التصنيفات
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# تحديد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحميل الموديل
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# تجهيز التحويلات
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    predicted_class = class_names[top_idx.item()]
    return predicted_class, top_prob.item()

# مثال على التوقع
test_image = r"C:\Users\abdo\Desktop\000006.jpg" # مسار الصورة
pred_class, prob = predict_image(test_image)

print(f"🔍 التوقع: {pred_class} — {prob:.3f}")
