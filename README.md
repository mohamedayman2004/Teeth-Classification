# 🦷 Teeth Classification — CNN Model

This project is part of my internship training, aiming to build a **robust computer vision model** to classify dental images into **7 distinct categories**.

---

## 📌 Project Goals

- ✅ **Preprocessing:** Normalize, augment, and prepare dental images for training.
- ✅ **Visualization:** Understand the class distribution and display examples before & after augmentation.
- ✅ **Model Building:** Develop a custom **CNN architecture** using TensorFlow.
- ✅ **Baseline Training:** Train the model, tune hyperparameters, and evaluate its performance.
- ✅ **Deployment Ready:** Store results & models for future improvement.

---

## 📁 Dataset Structure

The dataset is divided into:
- **Training**: 3087 images in 7 class folders.
- **Validation**: 1028 images in 7 class folders.
- **Testing**: 1028 images in 7 class folders.


---

## ⚙️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- Visual Studio Code
- Matplotlib & Seaborn (for visualization)


---

## Set Dataset Paths

- train_dir = '/content/drive/MyDrive/teeth_project/train'
- val_dir = '/content/drive/MyDrive/teeth_project/val'
- test_dir = '/content/drive/MyDrive/teeth_project/test'


---

## Preprocessing

-Normalization [0,1]
-Augmentation 


---

## 🧩 Model Architecture

The model was designed and built using TensorFlow and Keras:


```python
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 7

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()
```
## 📈 CNN Performance

| Metric              | Value   |
|---------------------|---------|
| Training Accuracy   | 82.1%   |
| Validation Accuracy | 81.8%   |
| Training Loss       | 0.52    |
| Validation Loss     | 0.54    |

## 🏆 ResNet50 Performance

| Metric              | Value    |
|---------------------|----------|
| Training Accuracy   | 99.02%   |
| Validation Accuracy | 98.83%   |
| Training Loss       | 0.0519   |
| Validation Loss     | 0.0340   |
| Learning Rate       | 1e-5     |





