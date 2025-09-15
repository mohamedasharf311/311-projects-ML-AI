# ✅ 1. تحميل المكتبات
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from google.colab import files
import matplotlib.pyplot as plt
import json
import urllib.request

# ✅ 2. تحميل أسماء الفئات (ImageNet)
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")

with open("imagenet_classes.txt") as f:
    categories = [s.strip() for s in f.readlines()]

# ✅ 3. تحميل نموذج ResNet18 المدرب مسبقًا
model = models.resnet18(pretrained=True)
model.eval()

# ✅ 4. تجهيز التحويلات المناسبة للنموذج
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

# ✅ 5. رفع صورة من الموبايل
uploaded = files.upload()

for file_name in uploaded.keys():
    # فتح الصورة
    image = Image.open(file_name).convert('RGB')
    plt.imshow(image)
    plt.axis('off')
    plt.title("📸 الصورة المختارة")
    plt.show()

    # تحويلها إلى Tensor
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    # التنبؤ
    with torch.no_grad():
        out = model(batch_t)
        _, index = torch.max(out, 1)

    prediction = categories[index[0]]
    print(f"✅ التنبؤ: {prediction}")