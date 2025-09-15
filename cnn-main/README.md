CIFAR-10 Image Classification with CNN

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset.


---

📌 Project Overview

Dataset: CIFAR-10 (60,000 images of size 32×32 in 10 classes).

Model: A simple CNN with convolutional, pooling, and dense layers.

Goal: Classify images into one of the 10 categories:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.




---

⚙️ Requirements

Make sure you have the following installed:

pip install tensorflow scikit-learn matplotlib numpy


---

🚀 How to Run

1. Clone this repository or copy the script.


2. Run the Python file:

python cifar10_cnn.py


3. The model will:

Load and preprocess CIFAR-10.

Train for 10 epochs.

Evaluate on the test set.

Show a random test image with the predicted label.





---

🧠 Model Architecture

Conv2D (32 filters, 3×3, ReLU)

MaxPooling2D (2×2)

Conv2D (64 filters, 3×3, ReLU)

MaxPooling2D (2×2)

Conv2D (64 filters, 3×3, ReLU)

Flatten

Dense (64, ReLU)

Dense (10 output neurons)



---

📊 Training & Evaluation

Loss function: SparseCategoricalCrossentropy

Optimizer: Adam

Metrics: Accuracy

Achieves around 70–75% accuracy after 10 epochs (may vary).



---

📷 Example Output

After training, the script shows a random test image with its predicted class:

Predicted: cat




---

🔮 Future Improvements

Add data augmentation to improve accuracy.

Use dropout layers to reduce overfitting.

Increase epochs or tune learning rate.

Experiment with more advanced architectures (ResNet, VGG, etc.).

