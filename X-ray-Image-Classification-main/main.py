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

# إعداد البيانات
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

# Prefetch لتحسين الأداء
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# بناء النموذج
base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # نجمد طبقات MobileNetV2

inputs = tf.keras.Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = Model(inputs, outputs)

# إعداد التدريب
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# تدريب النموذج
history = model.fit(train_ds, validation_data=val_ds, epochs=3)

# حفظ النموذج
model.save("xray_mobilenetv2_model.keras")

# التنبؤ على صورة من الجهاز
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

# مصفوفة الالتباس
y_true = []
y_pred = []
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

# تحويل النموذج إلى TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('xray_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\nTFLite model saved as 'xray_model.tflite'")
