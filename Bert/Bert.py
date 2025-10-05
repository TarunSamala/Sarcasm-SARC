import os
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Configuration
MAX_LEN = 50
BATCH_SIZE = 16
EPOCHS = 15
OUTPUT_DIR = "Sarcasm_outputs"
MODEL_NAME = "distilbert-base-uncased"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path, sample_size=100000):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(sample_size, random_state=42)
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    return df

# Main processing
df = load_data('../train-balanced-sarcasm.csv')
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['cleaned_comment'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# Class weight balancing
classes = np.unique(train_labels)
class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
class_weights = np.clip(class_weights, 0.5, 2)  # Limit extreme weights
class_weights_dict = dict(zip(classes, class_weights))
sample_weights = np.array([class_weights_dict[label] for label in train_labels])

# Initialize BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        max_length=MAX_LEN,
        truncation=True,
        padding=True,  # Dynamic padding
        return_tensors='tf'
    )

# Encode datasets
train_encodings = encode_texts(train_texts)
test_encodings = encode_texts(test_texts)

# Create TensorFlow datasets with sample weights
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask']
    },
    np.array(train_labels),
    sample_weights
)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask']
    },
    np.array(test_labels)
)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Enhanced DistilBERT model with regularization
def build_bert_model():
    config = DistilBertConfig.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        dropout=0.4,  # Increased dropout
        attention_dropout=0.3,  # Increased attention dropout
        seq_classif_dropout=0.5
    )
    
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    
    # Enhanced classifier with regularization
    l2_reg = tf.keras.regularizers.l2(0.02)
    model.classifier = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=l2_reg,
                            activity_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid',
                            kernel_regularizer=l2_reg)
    ])
    
    # Optimized optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=2e-5,
        weight_decay=0.03,
        clipnorm=1.0  # Gradient clipping
    )
    
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# Enhanced callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2,
                 min_delta=0.005, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=1, min_lr=1e-6)
]

# Training
model = build_bert_model()
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Visualization (same format)
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
plt.savefig(os.path.join(OUTPUT_DIR, 'Training_curves.png'), dpi=300)
plt.close()

# Generate predictions
raw_preds = model.predict(test_dataset)
y_pred = (raw_preds.logits > 0.5).astype(int)

# Classification Report
report = classification_report(test_labels, y_pred, 
                              target_names=['Non-Sarcastic', 'Sarcastic'])
with open(os.path.join(OUTPUT_DIR, 'Classification_report.txt'), 'w') as f:
    f.write("DistilBERT Classification Report:\n")
    f.write(report)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Non-Sarcastic', 'Sarcastic'],
           yticklabels=['Non-Sarcastic', 'Sarcastic'],
           annot_kws={"size": 22 })
plt.title('Confusion Matrix',fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.xlabel('Predicted Label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(os.path.join(OUTPUT_DIR, 'Confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")