import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MAX_LEN = 50
VOCAB_SIZE = 20000
EMBEDDING_DIM = 300
BATCH_SIZE = 64
EPOCHS = 40
OUTPUT_DIR = "Sarcasm_outputs"
GLOVE_PATH = "../../../glove.840B.300d.pkl"
DATA_PATH = "../../../train-balanced-sarcasm.csv"

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s!?\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data
def load_data():
    df = pd.read_csv(DATA_PATH, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(100000, random_state=42)
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    return df

# Load GloVe embeddings
def load_glove():
    with open(GLOVE_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

# Dataset pipeline
def create_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices((X, y)) \
           .batch(BATCH_SIZE) \
           .prefetch(tf.data.AUTOTUNE)

# Build model with regularization
def build_model(embedding_matrix):
    inputs = Input(shape=(MAX_LEN,))
    
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM,
                embeddings_initializer=Constant(embedding_matrix),
                trainable=True,
                mask_zero=True)(inputs)
    
    x = SpatialDropout1D(0.4)(x)
    
    x = Bidirectional(LSTM(128, return_sequences=True,
                         kernel_regularizer=regularizers.l2(1e-4),
                         recurrent_regularizer=regularizers.l2(1e-4)))(x)
    x = Bidirectional(LSTM(64,
                         kernel_regularizer=regularizers.l2(1e-4)))(x)
    
    x = Dense(128, activation='relu',
             kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.6)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Main execution
if __name__ == "__main__":
    # Load and process data
    df = load_data()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned_comment'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    
    # Sequencing
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), 
                           maxlen=MAX_LEN, padding='post')
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts),
                          maxlen=MAX_LEN, padding='post')

    # Load embeddings
    print("Loading GloVe embeddings...")
    glove_embeddings = load_glove()
    
    # Create embedding matrix
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i < VOCAB_SIZE and word in glove_embeddings:
            embedding_matrix[i] = glove_embeddings[word]

    # Class weights
    class_weights = compute_class_weight('balanced',
                                        classes=np.unique(train_labels),
                                        y=train_labels)
    class_weights = {i:w for i,w in enumerate(class_weights)}

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    # Build and train model
    model = build_model(embedding_matrix)
    history = model.fit(
        create_dataset(X_train, train_labels),
        validation_data=create_dataset(X_test, test_labels),
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves-glove-300d.png'), dpi=300)
    plt.close()

    # Generate reports
    y_pred = (model.predict(X_test, batch_size=BATCH_SIZE) > 0.5).astype(int)
    
    report = classification_report(test_labels, y_pred,
                                  target_names=['Non-Sarcastic', 'Sarcastic'])
    with open(os.path.join(OUTPUT_DIR, 'classification_report-glove-300d.txt'), 'w') as f:
        f.write(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(test_labels, y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix-glove-300d.png'), dpi=300)
    plt.close()

    print("Training completed. Outputs saved to:", OUTPUT_DIR)