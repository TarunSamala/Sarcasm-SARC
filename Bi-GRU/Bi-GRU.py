import os
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
MAX_LEN = 50  # Sequence length set for Reddit comments
VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
BATCH_SIZE = 256
EPOCHS = 30
OUTPUT_DIR = "Sarcasm_outputs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data
def load_data(file_path, sample_size=100000):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(sample_size, random_state=42)
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    return df

# Main processing: load the dataset and split into train and test sets.
df = load_data('../../train-balanced-sarcasm.csv')
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned_comment'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(train_texts)

# Sequence padding
X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), 
                        maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), 
                       maxlen=MAX_LEN, padding='post')

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Enhanced Model Architecture: Bidirectional GRU network
def build_bigru():
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding layer with L2 regularization
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  embeddings_regularizer=regularizers.l2(1e-4))(inputs)
    
    # SpatialDropout to combat overfitting
    x = SpatialDropout1D(0.5)(x)
    
    # First Bidirectional GRU layer with return_sequences=True
    x = Bidirectional(GRU(48, 
                          return_sequences=True, 
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4)))(x)
    
    # Second Bidirectional GRU layer (no return_sequences)
    x = Bidirectional(GRU(32))(x)
    
    # Dense layer for further processing with regularization
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.7)(x)
    
    # Output layer with sigmoid activation for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with RMSprop optimizer and binary cross-entropy loss
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Modified Callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, min_delta=0.002, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
]

# Build and train the Bi-GRU model
model = build_bigru()
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save training curves
plt.figure(figsize=(15, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves', pad=10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.4, 1.0)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves', pad=10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bigru_training_curves.png'), dpi=300)
plt.close()

# Generate predictions on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Classification Report saved to a text file
report = classification_report(test_labels, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'])
with open(os.path.join(OUTPUT_DIR, 'bigru_classification_report.txt'), 'w') as f:
    f.write("Bi-GRU Classification Report:\n")
    f.write(report)

# Confusion Matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('Bi-GRU Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'bigru_confusion_matrix.png'), dpi=300)
plt.close()

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
