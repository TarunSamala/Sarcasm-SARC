import os
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
MAX_LEN = 40  # Reduced sequence length
VOCAB_SIZE = 10000  # Reduced vocabulary size
EMBEDDING_DIM = 300
BATCH_SIZE = 32  # Small batch size
EPOCHS = 50
OUTPUT_DIR = "Sarcasm_outputs"
FASTTEXT_PATH = "../../../crawl-300d-2M.vec"

# Configure GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]
        )
    except RuntimeError as e:
        print(e)

# Mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data with limited samples
def load_data(file_path):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(80000, random_state=42)  # Reduced sample size
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    return df

# Dataset pipeline
def create_dataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# Load and process data
df = load_data('../../../train-balanced-sarcasm.csv')
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned_comment'], 
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

# Sequence padding
X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), 
                       maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts),
                      maxlen=MAX_LEN, padding='post', truncating='post')

# Load FastText embeddings
def load_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

print("Loading embeddings...")
embeddings = load_embeddings(FASTTEXT_PATH)

# Build embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE and word in embeddings:
        embedding_matrix[i] = embeddings[word]

# Memory-efficient model with type annotations
def build_model():
    inputs = Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_layer')
    
    # Embedding layer with explicit dtype
    x = Embedding(
        VOCAB_SIZE, EMBEDDING_DIM,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
        mask_zero=True,
        dtype=tf.float32  # Explicit dtype for embedding layer
    )(inputs)
    
    # Bi-LSTM layer with mixed precision policy
    x = Bidirectional(LSTM(64, return_sequences=False), dtype=tf.float16)(x)
    
    # Dense layers with proper dtype casting
    x = Dense(64, activation='relu', dtype=tf.float16)(x)
    x = Dropout(0.4)(x)
    
    # Output layer with fixed float32 dtype
    outputs = Dense(1, activation='sigmoid', dtype=tf.float32)(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# Convert to TF Dataset
train_dataset = create_dataset(X_train, train_labels)
test_dataset = create_dataset(X_test, test_labels)

# Train model
model = build_model()
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save results
def save_results():
    # Training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()

    # Classification report
    y_pred = (model.predict(X_test, batch_size=BATCH_SIZE) > 0.5).astype(int)
    report = classification_report(test_labels, y_pred)
    with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

save_results()
print("Process completed successfully!")