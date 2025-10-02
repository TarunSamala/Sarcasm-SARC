import os
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertConfig

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Create output directory
OUTPUT_DIR = 'sarcasm_outputs_hybrid_reddit_fixed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path, sample_size=200000):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    if sample_size is not None:
        df = df.sample(sample_size, random_state=42)
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    return df

# Focal Loss implementation
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, self.gamma)
        bce = -self.alpha * y_true * tf.math.log(y_pred) - (1 - self.alpha) * (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(focal_factor * bce)

# Configuration
MAX_LENGTH = 50
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5

if __name__ == "__main__":
    # Load and split data
    df = load_data('../train-balanced-sarcasm.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_comment'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # Class weights with smoothing (not used since balanced, but computed)
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = np.clip(class_weights, 0.5, 2)
    class_weights_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weights_dict[label] for label in y_train])

    # Tokenization
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(
        X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )
    test_encodings = tokenizer(
        X_test.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='tf'
    )

    # Dataset prep
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': train_encodings['input_ids'],
         'attention_mask': train_encodings['attention_mask']},
        y_train
    )).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': test_encodings['input_ids'],
         'attention_mask': test_encodings['attention_mask']},
        y_test
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Load DistilBERT base model for embeddings
    bert_model = TFDistilBertModel.from_pretrained(
        'distilbert-base-uncased',
        from_pt=False
    )

    # Unfreeze last 4 layers for better fine-tuning
    for layer in bert_model.layers:
        layer.trainable = False
    for layer in bert_model.distilbert.transformer.layer[-4:]:
        layer.trainable = True

    # Build hybrid model: BERT + CNN + BiLSTM + Attention
    inputs = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')

    # BERT embeddings
    bert_outputs = bert_model(inputs, attention_mask=attention_mask)
    embeddings = bert_outputs.last_hidden_state  # (batch, seq_len, 768)

    # CNN branch
    cnn = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(embeddings)
    cnn = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn)
    cnn = tf.keras.layers.GlobalMaxPooling1D()(cnn)  # (batch, 64)

    # Expand CNN output for sequence compatibility
    cnn_expanded = tf.keras.layers.RepeatVector(MAX_LENGTH)(cnn)  # (batch, seq_len, 64)
    cnn_expanded = tf.keras.layers.Reshape((MAX_LENGTH, 64))(cnn_expanded)

    # Concatenate with BERT embeddings
    fused = tf.keras.layers.Concatenate(axis=-1)([embeddings, cnn_expanded])  # (batch, seq_len, 832)

    # BiLSTM
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.0),
        name='bilstm'
    )(fused)  # (batch, seq_len, 256)

    # Attention
    attention = tf.keras.layers.Attention()([bilstm, bilstm])  # (batch, seq_len, 256)
    attention = tf.keras.layers.GlobalAveragePooling1D()(attention)  # (batch, 256)

    # Classification head
    l2_reg = tf.keras.regularizers.l2(0.03)
    dropout = tf.keras.layers.Dropout(0.5)(attention)
    dense = tf.keras.layers.Dense(
        64, 
        activation='relu', 
        kernel_regularizer=l2_reg
    )(dropout)
    dropout2 = tf.keras.layers.Dropout(0.3)(dense)
    outputs = tf.keras.layers.Dense(1, activation=None, name='outputs')(dropout2)  # Pseudo-probs

    model = tf.keras.Model(inputs=[inputs, attention_mask], outputs=outputs)

    # Optimizer and loss
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        clipnorm=1.0
    )
    loss = FocalLoss(gamma=2.0, alpha=0.5)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        min_delta=0.005,
        restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )

    # Train
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    # Save training curves
    def save_training_curves(history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy Curves')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), bbox_inches='tight')
        plt.close()

    save_training_curves(history)

    # Predictions (FIX: Treat outputs as probabilities, no extra sigmoid)
    outputs = model.predict(test_dataset)  # These are pseudo-probabilities
    probabilities = outputs.flatten()
    y_pred = (probabilities > 0.5).astype(int)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'])
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write("Hybrid Model Classification Report:\n")
        f.write(report)

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Non-Sarcastic', 'Sarcastic'],
        yticklabels=['Non-Sarcastic', 'Sarcastic'],
        annot_kws={"size": 22}
    )
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    print(f"All results saved to '{os.path.abspath(OUTPUT_DIR)}'")