#https://www.kaggle.com/datasets/evilspirit05/ecg-analysis/data

# ===============================
# IMPORT REQUIRED LIBRARIES
# ===============================

# Basic libraries
import os
import random
import numpy as np
import pandas as pd
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Image processing
from PIL import Image

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


from tensorflow import keras
from tensorflow.keras import layers, models

from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Warnings
import warnings
warnings.filterwarnings("ignore")

print("All libraries imported successfully!")

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ===============================
# EXPLORE DATASET: DATASET PATH & CLASS LABELS
# ===============================


def create_dataframe(root_dir):
    data = []

    for label in os.listdir(root_dir):
        class_path = os.path.join(root_dir, label)

        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            data.append([img_path, label])

    df = pd.DataFrame(data, columns=["filepaths", "labels"])
    return df

test_dir = "Data/test"
train_dir = "Data/train"

df1 = create_dataframe(test_dir)
df2 = create_dataframe(train_dir)
df=combined_df = pd.concat([df1, df2], ignore_index=True)

print(df.head())

print(df['labels'].unique())

print(df['labels'].value_counts())

# ===============================
# DATASET CLASS DISTRIBUTION
# ===============================


sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x="labels", palette="viridis", ax=ax)

ax.set_title("Distribution of Target Class", fontsize=14, fontweight='bold')
ax.set_xlabel("Type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=11, color='black', 
                xytext=(0, 5), textcoords='offset points')

plt.xticks(rotation = 0)
plt.savefig("FClasscountbar.png")

plt.show()

label_counts = df["labels"].value_counts()

fig, ax = plt.subplots(figsize=(8, 6))
colors = sns.color_palette("viridis", len(label_counts))

ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
       startangle=140, colors=colors, textprops={'fontsize': 12, 'weight': 'bold'},
       wedgeprops={'edgecolor': 'black', 'linewidth': 1})

ax.set_title("Distribution of Target Class - Pie Chart", fontsize=14, fontweight='bold')
plt.savefig("Fclassdistpie.png")

plt.show()

# ===============================
# Data Generators
# ===============================

# Image data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# ===============================
# Create Winograd Model with CNN
# ===============================

def create_winograd_cnn(input_shape, num_classes):
    model = models.Sequential([
        # Winograd is typically applied to 3x3 convolutions
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation = 'softmax')
    
    ])
    return model


input_shape = (150, 150, 3)
num_classes = 4
model = create_winograd_cnn(input_shape, num_classes)
# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy * 100:.2f}%')

model.save("winoecg.h5")

plt.plot(history.history["loss"],label="Train Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.legend()
plt.savefig("Floss.png")

plt.show()

plt.plot(history.history["accuracy"],label="Train accuracy")
plt.plot(history.history["val_accuracy"],label="Validation accuracy")
plt.legend()
plt.savefig("Facc.png")

plt.show()

