import os
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split  
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MAX_LEN = 70  # Optimized for Reddit comments
VOCAB_SIZE = 30000
EMBEDDING_DIM = 100
BATCH_SIZE = 512
EPOCHS = 40
OUTPUT_DIR = "Sarcasm_outputs"
GLOVE_PATH = "../../../glove.6B.100d.txt"  # Update with your path

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enhanced text cleaning for Reddit comments
def clean_reddit_text(text):
    text = str(text).lower()
    # Remove markdown URLs
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^a-zA-Z\s!?\'.,]', '', text)
    # Handle repeated punctuation
    text = re.sub(r'(!|\?|\.){2,}', r'\1', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data
def load_reddit_data(file_path, sample_size=150000):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(sample_size, random_state=42)
    df['cleaned_comment'] = df['comment'].apply(clean_reddit_text)
    return df

# Load dataset
df = load_reddit_data('../../../train-balanced-sarcasm.csv')

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(df['cleaned_comment'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Tokenization and sequencing
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), 
                        maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), 
                       maxlen=MAX_LEN, padding='post', truncating='post')

# Load GloVe embeddings
def load_glove(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

embeddings = load_glove(GLOVE_PATH)

# Create embedding matrix with trainable embeddings
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE and word in embeddings:
        embedding_matrix[i] = embeddings[word]

# Enhanced Bi-LSTM Model with regularization
def build_reddit_model():
    inputs = Input(shape=(MAX_LEN,))
    
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  embeddings_initializer=Constant(embedding_matrix),
                  mask_zero=True,
                  trainable=True)(inputs)  # Allow fine-tuning
    
    x = SpatialDropout1D(0.4)(x)
    
    x = Bidirectional(LSTM(128, return_sequences=True,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4)))(x)
    
    x = Bidirectional(LSTM(64,
                          kernel_regularizer=regularizers.l2(1e-4)))(x)
    
    x = Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.6)(x)
    x = Dense(64, activation='relu',
              kernel_regularizer=regularizers.l2(1e-4))(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Class weighting for imbalance
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(train_labels), 
                                     y=train_labels)
class_weights = {i:w for i,w in enumerate(class_weights)}

# Enhanced callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8,
                 min_delta=0.002, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=4, min_lr=1e-6),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.h5'),
                   save_best_only=True, monitor='val_accuracy')
]

# Convert labels to arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Training
model = build_reddit_model()
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Save training curves
def save_training_curves(history):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Accuracy Curves', fontsize=14)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0.5, 1.0)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Loss Curves', fontsize=14)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300)
    plt.close()

save_training_curves(history)

# Generate predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Save classification report
report = classification_report(test_labels, y_pred, 
                               target_names=['Non-Sarcastic', 'Sarcastic'])
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Reddit Sarcasm Classification Report:\n")
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('Reddit Sarcasm Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()

# Final evaluation
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")