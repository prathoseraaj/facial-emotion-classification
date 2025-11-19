import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

CONFIG = {
    'dataset_path': 'Dataset(images)',  # Path to your images folder
    'model_path': './facial_expression_model.h5',
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'emotion_mapping': {
        'HA': 'Happy',
        'SA': 'Sad',
        'AN': 'Angry',
        'NE': 'Neutral',
        'FE': 'Fear',
        'DI': 'Disgust',
        'SU': 'Surprise'
    }
}

# Data Loading & Preprocessing 
def load_dataset(dataset_path):
    """
    Load images from dataset folder and extract labels from filenames
    Expected format: 23BTRCL139-01-HA-01.jpg
    """
    images = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return None, None
    
    image_files = list(Path(dataset_path).glob('*.jpg')) + list(Path(dataset_path).glob('*.png'))
    print(f"Found {len(image_files)} images")
    
    for img_path in image_files:
        try:
            # Extract emotion code from filename
            filename = img_path.stem  # Gets filename without extension
            parts = filename.split('-')
            
            if len(parts) >= 3:
                emotion_code = parts[2]  # HA, SA, AN, etc.
                
                # Read and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Convert BGR to RGB and resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, CONFIG['img_size'])
                img = img / 255.0  # Normalize to [0, 1]
                
                images.append(img)
                labels.append(emotion_code)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Successfully loaded {len(images)} images")
    print(f"Label distribution: {np.unique(labels, return_counts=True)}")
    
    return images, labels

# Model Architecture 
def build_model(num_classes):
    """
    Build CNN model with transfer learning (MobileNetV2 base)
    Lightweight and fast for real-time inference
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights initially
    base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=(*CONFIG['img_size'], 3)),
        
        # Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Preprocessing for MobileNetV2
        layers.Lambda(keras.applications.mobilenet_v2.preprocess_input),
        
        # Base model
        base_model,
        
        # Custom head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Training
def train_model(images, labels):
    """
    Train the facial expression recognition model
    """
    # Encode labels to integers
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = keras.utils.to_categorical(labels_encoded, len(le.classes_))
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_categorical,
        test_size=CONFIG['validation_split'],
        random_state=42,
        stratify=labels_encoded
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Build and compile model
    model = build_model(num_classes=len(le.classes_))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and label encoder
    model.save(CONFIG['model_path'])
    with open('./label_encoder.json', 'w') as f:
        json.dump({'classes': le.classes_.tolist()}, f)
    
    print(f"Model saved to {CONFIG['model_path']}")
    return model, le, history

# REAL-TIME INFERENCE 
def run_webcam_inference(model, label_encoder):
    """
    Real-time facial expression detection from webcam
    """
    # Load face cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    print("Starting webcam... Press 'q' to quit")
    
    # Smoothing for predictions
    prediction_history = []
    history_size = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, fw, fh) in faces:
            # Extract face region
            face_roi = frame[y:y+fh, x:x+fw]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, CONFIG['img_size'])
            face_normalized = face_resized / 255.0
            
            # Predict
            face_batch = np.expand_dims(face_normalized, axis=0)
            predictions = model.predict(face_batch, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            confidence = predictions[emotion_idx]
            emotion_code = label_encoder.classes_[emotion_idx]
            emotion_name = CONFIG['emotion_mapping'].get(emotion_code, emotion_code)
            
            # Smooth predictions
            prediction_history.append(emotion_name)
            if len(prediction_history) > history_size:
                prediction_history.pop(0)
            smoothed_emotion = max(set(prediction_history), key=prediction_history.count)
            
            # Draw rectangle and label
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
            
            label_text = f"{smoothed_emotion} ({confidence:.2f})"
            cv2.putText(frame, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display frame
        cv2.imshow('Facial Expression Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# MAIN EXECUTION 
def main():
    """
    Main pipeline orchestrator
    """
    print("=" * 60)
    print("FACIAL EXPRESSION RECOGNITION SYSTEM")
    print("=" * 60)
    
    # Check if model exists
    if os.path.exists(CONFIG['model_path']):
        print("\nExisting model found!")
        choice = input("Do you want to (1) Train new model or (2) Use existing model? Enter 1 or 2: ")
        
        if choice == '2':
            print("Loading existing model...")
            model = keras.models.load_model(CONFIG['model_path'])
            with open('./label_encoder.json', 'r') as f:
                le_data = json.load(f)
                le = LabelEncoder()
                le.classes_ = np.array(le_data['classes'])
            run_webcam_inference(model, le)
            return
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    images, labels = load_dataset(CONFIG['dataset_path'])
    
    if images is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Train model
    print("\n[2/3] Training model...")
    model, le, history = train_model(images, labels)
    
    # Plot training history
    print("\n[3/3] Plotting training history...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./training_history.png')
    print("Training history saved to training_history.png")
    
    # Run webcam inference
    print("\nStarting real-time inference...")
    run_webcam_inference(model, le)

if __name__ == "__main__":
    main()