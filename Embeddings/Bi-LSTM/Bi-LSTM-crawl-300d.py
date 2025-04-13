import os
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
MAX_LEN = 50                # Maximum sequence length for comments
VOCAB_SIZE = 20000          # Maximum vocabulary size
EMBEDDING_DIM = 300         # Using 300-dimensional embeddings from crawl-300d-2M.vec
BATCH_SIZE = 256
EPOCHS = 30
OUTPUT_DIR = "Sarcasm_outputs"
EMBEDDING_PATH = '../../../crawl-300d-2M.vec'  # Path to your crawl-300d-2M.vec embeddings file
SAMPLE_SIZE = 1000000       # Use 1 million comments from the dataset

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data by sampling SAMPLE_SIZE rows from the CSV
def load_data(file_path, sample_size=SAMPLE_SIZE):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(sample_size, random_state=42)
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    return df

# Load dataset (adjust the CSV file path if needed)
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

# Load the crawl-300d-2M.vec embeddings.
print("Loading crawl-300d-2M embeddings...")
embeddings_index = {}
with open(EMBEDDING_PATH, encoding='utf8') as f:
    # Check if the first line is a header (contains two numbers)
    first_line = f.readline().strip().split()
    if len(first_line) == 2:
        # Header exists; skip it
        print("Skipping header line...")
    else:
        # Not a header; process the first line
        word = first_line[0]
        coefs = np.asarray(first_line[1:], dtype='float32')
        embeddings_index[word] = coefs
    # Process the remainder of the file
    for line in f:
        values = line.rstrip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print("Total embeddings loaded:", len(embeddings_index))

# Build the embedding matrix for our vocabulary.
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Define the Bidirectional LSTM model using crawl embeddings.
def build_bilstm_with_crawl():
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  trainable=True,
                  embeddings_regularizer=regularizers.l2(1e-4))(inputs)
    # Use a moderate spatial dropout
    x = SpatialDropout1D(0.3)(x)
    # First Bi-LSTM layer with increased units; returns sequences.
    x = Bidirectional(LSTM(128, return_sequences=True,
                           kernel_regularizer=regularizers.l2(1e-4),
                           recurrent_regularizer=regularizers.l2(1e-4)))(x)
    # Second Bi-LSTM layer (does not return sequences)
    x = Bidirectional(LSTM(64,
                           kernel_regularizer=regularizers.l2(1e-4),
                           recurrent_regularizer=regularizers.l2(1e-4)))(x)
    # Fully connected layer to learn complex features
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.4)(x)
    # Output layer with sigmoid activation for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Define callbacks: early stopping and learning rate reduction.
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
]

# Build and train the model.
model = build_bilstm_with_crawl()
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save training curves (accuracy and loss plots) to the specified filename.
plt.figure(figsize=(15, 6))
# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves', pad=10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.4, 1.0)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()
# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves', pad=10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
train_curve_path = os.path.join(OUTPUT_DIR, 'training_curves-crawl_BiLSTM.png')
plt.savefig(train_curve_path, dpi=300)
plt.close()

# Generate predictions on the test set.
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Save the classification report.
report = classification_report(test_labels, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'])
report_path = os.path.join(OUTPUT_DIR, 'crawl_BiLSTM_classification_report.txt')
with open(report_path, 'w') as f:
    f.write("BiLSTM with crawl-300d-2M.vec Classification Report:\n")
    f.write(report)

# Plot and save the confusion matrix.
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('BiLSTM with crawl-300d-2M.vec Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
conf_mat_path = os.path.join(OUTPUT_DIR, 'crawl_BiLSTM_confusion_matrix.png')
plt.savefig(conf_mat_path, dpi=300)
plt.close()

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
