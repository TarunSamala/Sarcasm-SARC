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
MAX_LEN = 50               # Sequence length for the comments
VOCAB_SIZE = 20000         # Maximum vocabulary size
EMBEDDING_DIM = 100        # Using 100-dimensional GloVe embeddings
BATCH_SIZE = 256
EPOCHS = 30
OUTPUT_DIR = "Sarcasm_outputs"
GLOVE_PATH = '../../../glove.6B.100d.txt'  # Path to the GloVe file
SAMPLE_SIZE = 1000000      # Use 1 million comments from the dataset

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data, sampling SAMPLE_SIZE rows from the CSV
def load_data(file_path, sample_size=SAMPLE_SIZE):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(sample_size, random_state=42)
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    return df

# Load dataset (replace '../../train-balanced-sarcasm.csv' with the correct path if needed)
df = load_data('../../../train-balanced-sarcasm.csv', sample_size=SAMPLE_SIZE)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned_comment'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Tokenization and sequence padding
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts),
                        maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts),
                       maxlen=MAX_LEN, padding='post')

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Load the GloVe embeddings and create an embedding matrix
print("Loading GloVe embeddings...")
embeddings_index = {}
with open(GLOVE_PATH, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Define the Bidirectional GRU model with pre-trained GloVe embeddings
def build_bigru_with_glove():
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding layer with pre-trained GloVe weights; trainable set to True to allow fine-tuning.
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  trainable=True,
                  embeddings_regularizer=regularizers.l2(1e-4))(inputs)
    
    # Use a slightly lower dropout rate to preserve more information during training.
    x = SpatialDropout1D(0.4)(x)
    
    # Increase model capacity by upping GRU units.
    x = Bidirectional(GRU(64, return_sequences=True,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4)))(x)
    
    x = Bidirectional(GRU(48,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4)))(x)
    
    # Expand Dense layer for improved learning
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Setup callbacks for early stopping and adaptive learning rate
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
]

# Build and train the model
model = build_bigru_with_glove()
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save training curves: accuracy and loss
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
plt.savefig(os.path.join(OUTPUT_DIR, 'bigru_glove_1M_training_curves.png'), dpi=300)
plt.close()

# Generate predictions on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Save classification report
report = classification_report(test_labels, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'])
with open(os.path.join(OUTPUT_DIR, 'bigru_glove_1M_classification_report.txt'), 'w') as f:
    f.write("Bi-GRU with GloVe on 1M Comments Classification Report:\n")
    f.write(report)

# Plot and save the confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('Bi-GRU with GloVe on 1M Comments Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'bigru_glove_1M_confusion_matrix.png'), dpi=300)
plt.close()

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
