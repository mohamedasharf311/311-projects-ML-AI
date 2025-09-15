MNIST Handwritten Digit Classification with CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset.


---

📌 Project Overview

Dataset: MNIST (60,000 training images, 10,000 testing images).

Task: Classify 28×28 grayscale images into 10 categories (digits 0–9).

Model: A simple CNN with convolutional, pooling, and dense layers.



---

⚙️ Requirements

Install the dependencies before running:

pip install tensorflow numpy matplotlib


---

🚀 How to Run

1. Clone this repository or copy the script.


2. Run the Python file:

python mnist_cnn.py


3. The script will:

Load and preprocess the MNIST dataset.

Train a CNN for 5 epochs.

Evaluate accuracy on the test set.

Show a sample image with its predicted label.





---

🧠 Model Architecture

Conv2D (32 filters, 3×3, ReLU)

MaxPooling2D (2×2)

Conv2D (64 filters, 3×3, ReLU)

MaxPooling2D (2×2)

Conv2D (64 filters, 3×3, ReLU)

Flatten

Dense (64, ReLU)

Dense (10, Softmax)



---

📊 Training & Evaluation

Loss function: Sparse Categorical Crossentropy

Optimizer: Adam

Batch size: 64

Epochs: 5


Expected performance:

Test accuracy: ~98%


---

📷 Example Output

After training, the script displays a test image with the predicted label:

Predicted label: 7




---

🔮 Future Improvements

Increase epochs for better accuracy.

Add Dropout layers to prevent overfitting.

Try more advanced architectures (e.g., LeNet, ResNet).

Experiment with data augmentation.


