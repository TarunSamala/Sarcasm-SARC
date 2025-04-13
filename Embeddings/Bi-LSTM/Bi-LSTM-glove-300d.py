import os
import pickle
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
MAX_LEN = 50               # Maximum sequence length for comments
VOCAB_SIZE = 20000         # Maximum vocabulary size
EMBEDDING_DIM = 300        # Using 300-dimensional embeddings from glove.840B.300d.pkl
BATCH_SIZE = 256
EPOCHS = 30
OUTPUT_DIR = "Sarcasm_outputs"
GLOVE_PATH = '../../../glove.840B.300d.pkl'  # Path to the pickle file with GloVe embeddings
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

# Load the dataset (adjust path if necessary)
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

# Load the GloVe embeddings from pickle file (glove.840B.300d.pkl)
print("Loading GloVe embeddings from pickle file...")
with open(GLOVE_PATH, 'rb') as f:
    embeddings_index = pickle.load(f)

# Build the embedding matrix using the tokenizer's word index and the loaded embeddings
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Define the Bidirectional LSTM model with pre-trained GloVe embeddings.
# Here, we explicitly initialize the final Dense layer's kernel with very small values and the bias to zero so that
# initial outputs (via the sigmoid) are near 0.5 (chance level for a balanced binary task).
def build_bilstm_with_glove():
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding layer with pre-trained weights; trainable=True to allow fine-tuning.
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  trainable=True,
                  embeddings_regularizer=regularizers.l2(1e-4))(inputs)
    
    # SpatialDropout1D for regularization
    x = SpatialDropout1D(0.4)(x)
    
    # First Bidirectional LSTM layer with return_sequences=True
    x = Bidirectional(LSTM(64, return_sequences=True,
                           kernel_regularizer=regularizers.l2(1e-4),
                           recurrent_regularizer=regularizers.l2(1e-4)))(x)
    
    # Second Bidirectional LSTM layer
    x = Bidirectional(LSTM(48,
                           kernel_regularizer=regularizers.l2(1e-4),
                           recurrent_regularizer=regularizers.l2(1e-4)))(x)
    
    # Fully connected layer for further feature learning
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    
    # Final Dense layer: we set kernel initializer to RandomUniform with a very small range and bias to zeros
    outputs = Dense(1, activation='sigmoid',
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01),
                    bias_initializer=tf.keras.initializers.Zeros())(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Setup callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
]

# Build and train the Bidirectional LSTM model
model = build_bilstm_with_glove()
history = model.fit(
    X_train, train_labels,
    validation_data=(X_test, test_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save training curves (accuracy and loss plots) with the specified filename
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
train_curve_path = os.path.join(OUTPUT_DIR, 'training_curves-glove840_BiLSTM.png')
plt.savefig(train_curve_path, dpi=300)
plt.close()

# Generate predictions on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Save classification report
report = classification_report(test_labels, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'])
report_path = os.path.join(OUTPUT_DIR, 'glove840_BiLSTM_classification_report.txt')
with open(report_path, 'w') as f:
    f.write("BiLSTM with glove.840B.300d.pkl Classification Report:\n")
    f.write(report)

# Plot and save the confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('BiLSTM with glove.840B.300d.pkl Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
conf_mat_path = os.path.join(OUTPUT_DIR, 'glove840_BiLSTM_confusion_matrix.png')
plt.savefig(conf_mat_path, dpi=300)
plt.close()

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
