import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===== CONFIG =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

VOCAB_SIZE = 20000
MAX_LENGTH = 50
EMBEDDING_DIM = 128
BATCH_SIZE = 256
EPOCHS = 15
QUBITS = 4
Q_LAYERS = 2
OUTPUT_DIR = "Sarcasm_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== TEXT CLEANING =====
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===== DATA LOADING =====
def load_data(file_path, sample_size=100000):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    return df

# ===== QUANTUM CIRCUIT =====
dev = qml.device("default.qubit", wires=QUBITS)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(QUBITS))
    qml.templates.BasicEntanglerLayers(weights, wires=range(QUBITS))
    return [qml.expval(qml.PauliZ(w)) for w in range(QUBITS)]

weight_shapes = {"weights": (Q_LAYERS, QUBITS)}
qlayer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=QUBITS)

# ===== MODEL =====
def build_q_lstm_model():
    inputs = tf.keras.Input(shape=(MAX_LENGTH,))
    x = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
    x = tf.keras.layers.SpatialDropout1D(0.3)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.3)
    )(x)

    # Reduce to match QUBITS for quantum layer
    x = tf.keras.layers.Dense(QUBITS)(x)

    x = qlayer(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        metrics=["accuracy"]
    )
    return model

# ===== MAIN =====
if __name__ == "__main__":
    df = load_data("../train-balanced-sarcasm.csv")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_comment'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # Tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(X_train)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post')
    X_test_pad = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, min_delta=0.001, restore_best_weights=True
    )

    # Build & Train model
    model = build_q_lstm_model()
    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_test_pad, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )

    # ===== PLOTS =====
    plt.figure(figsize=(12, 4))
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

    plt.savefig(os.path.join(OUTPUT_DIR, "Training_curves.png"))
    plt.close()

    # ===== REPORTS =====
    y_pred = (model.predict(X_test_pad) > 0.5).astype(int).flatten()
    report = classification_report(y_test, y_pred, target_names=['Non-Sarcastic','Sarcastic'])
    with open(os.path.join(OUTPUT_DIR, "Classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=["Non-Sarcastic", "Sarcastic"],
                yticklabels=["Non-Sarcastic", "Sarcastic"],
                annot_kws={"size": 22})
    plt.xlabel("Predicted Label", fontsize=20)
    plt.ylabel("True Label", fontsize=20)
    plt.title("Confusion Matrix", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(OUTPUT_DIR, "Confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ All results saved to '{OUTPUT_DIR}' folder.")
