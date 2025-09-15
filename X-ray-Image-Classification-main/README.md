🩻 X-ray Classification using MobileNetV2

📌 Overview

This project trains a deep learning model to classify X-ray images using Transfer Learning with MobileNetV2.
It includes:

Data preprocessing

Model training & evaluation

Confusion matrix & classification report

Single image prediction

Conversion to TensorFlow Lite (TFLite) for deployment



---

📂 Dataset

The dataset should be organized in folders like this:

Xray_Data_Organized/
│── class_1/
│   ├── img1.jpg
│   ├── img2.jpg
│── class_2/
│   ├── img3.jpg
│   ├── img4.jpg
...

We split the dataset into training (80%) and validation (20%).


---

⚙️ Code Explanation

1. Import libraries

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from google.colab import files
from tensorflow.keras.preprocessing import image

We import TensorFlow, Keras, and additional libraries for visualization and evaluation.


---

2. Load dataset

dataset_path = "/content/drive/MyDrive/Xray_Data_Organized"
batch_size = 16
img_size = (160, 160)

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

Loads images from directories.

Splits data into training & validation.

Automatically assigns labels from folder names.



---

3. Optimize performance

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

Uses TensorFlow’s prefetch to load data faster during training.


---

4. Build the model (Transfer Learning)

base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze pretrained layers

inputs = tf.keras.Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = Model(inputs, outputs)

Loads MobileNetV2 pretrained on ImageNet.

Freezes base layers to keep pretrained features.

Adds custom layers for classification.



---

5. Compile the model

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Uses Adam optimizer with small learning rate.

Loss: Sparse categorical crossentropy.

Metric: Accuracy.



---

6. Train the model

history = model.fit(train_ds, validation_data=val_ds, epochs=3)

Trains for 3 epochs (can be increased).


---

7. Save the model

model.save("xray_mobilenetv2_model.keras")

Saves trained model in .keras format.


---

8. Predict a single image

uploaded = files.upload()
for fn in uploaded.keys():
    img_path = fn

img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]
confidence = 100 * np.max(pred)

plt.imshow(img)
plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
plt.axis("off")
plt.show()

Uploads an image.

Preprocesses it.

Predicts the class with confidence score.



---

9. Evaluate model performance

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

Creates confusion matrix.

Shows classification report (precision, recall, F1-score).



---

10. Convert model to TFLite

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('xray_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\nTFLite model saved as 'xray_model.tflite'")

Converts model to lightweight TFLite format.

Suitable for mobile/embedded devices.



---

📊 Results

Confusion matrix visualization.

Classification report with accuracy, precision, recall, F1-score.

Single image prediction with confidence score.



---

🚀 Deployment

.keras model can be reloaded with tf.keras.models.load_model.

.tflite model can be used for deployment on mobile apps.
