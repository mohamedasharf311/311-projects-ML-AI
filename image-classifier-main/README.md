# image-classifier
# 🧠 Image Classifier using PyTorch (ResNet18)

This project uses a pre-trained ResNet18 model from torchvision to classify real-world images (e.g. dog, cat, jeans, airplane...) using PyTorch and Google Colab.

## 🚀 Features
- Upload image from mobile or PC
- Auto prediction using ResNet18
- Works on Google Colab with GPU
- Supports 1000+ ImageNet classes

## 📸 Example
Input image:

![example](example.jpg)

Prediction: `jean` (pants)

## 🛠️ How to Use

1. Open `main.ipynb` in Google Colab
2. Run all cells
3. Upload an image when prompted
4. View the prediction result

## 🔧 Requirements

- torch
- torchvision
- matplotlib
- PIL
- Google Colab (for interface)

## 🧠 Model

We use `resnet18(pretrained=True)` from `torchvision.models`, trained on ImageNet.

---

## 🧑‍💻 Author

Mohamed Ashraf – [@your_username](https://github.com/your_username)