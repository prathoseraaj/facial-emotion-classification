import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
DATA_DIR = 'data/' # Your folder with 2000+ images
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# === 1. PARSE FILENAMES (Same as before) ===
label_map = {'HA': 'Happy', 'SA': 'Sad', 'AN': 'Angry', 'NE': 'Neutral', 
             'FE': 'Fear', 'DI': 'Disgust', 'SU': 'Surprise'}

filenames, categories = [], []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            parts = filename.split('-')
            code = parts[2]
            if code in label_map:
                filenames.append(filename)
                categories.append(label_map[code])
        except: pass

df = pd.DataFrame({'filename': filenames, 'category': categories})
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])

# === 2. ROBUST DATA GENERATOR ===
# ResNet expects specific preprocessing
from tensorflow.keras.applications.resnet_v2 import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Uses ResNet's own smart logic
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df, DATA_DIR, x_col='filename', y_col='category',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df, DATA_DIR, x_col='filename', y_col='category',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)

# === 3. THE "80% PRE-TRAINED" MODEL ===
# Load ResNet50V2 with weights trained on ImageNet (14 Million images)
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# FREEZE the base. This ensures we use the pre-trained "brain" (80%)
base_model.trainable = False 

# Add our custom head (The 20%)
x = base_model.output
x = GlobalAveragePooling2D()(x) # Smarter than Flatten() for variable images
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x) # Ignores noise/bad pixels
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === 4. TRAIN ONLY THE TOP LAYER ===
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training only the top layer (Adapting the 20%)...")
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

model.save('resnet_emotion_model.h5')
print("Model Saved.")