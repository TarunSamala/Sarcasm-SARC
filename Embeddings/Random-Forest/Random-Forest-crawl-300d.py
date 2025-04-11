import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure paths
OUTPUT_DIR = 'Sarcasm_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_fasttext(embeddings_path):
    """Load FastText embeddings from .vec file"""
    embeddings = {}
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def text_to_vector(text, embeddings):
    """Convert text to average FastText vector"""
    words = text.split()
    vectors = [embeddings.get(word, np.zeros(300)) for word in words]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

def main():
    # Load and preprocess data
    df = pd.read_csv('../../../train-balanced-sarcasm.csv', usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(100000, random_state=42)
    df['comment'] = df['comment'].str.lower().str.replace('[^\w\s]', '', regex=True)
    
    # Load FastText embeddings
    print("Loading FastText embeddings...")
    ft_embeddings = load_fasttext('../../../crawl-300d-2M.vec')
    
    # Convert text to vectors
    print("Converting comments to vectors...")
    X = np.array([text_to_vector(text, ft_embeddings) for text in df['comment']])
    y = df['label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Train and evaluate
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, 
                               class_weight='balanced',
                               n_jobs=-1,
                               random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Save classification report
    report = classification_report(y_test, y_pred)
    with open(f'{OUTPUT_DIR}/classification_report-crawl-300d.txt', 'w') as f:
        f.write(report)
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='viridis',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('FastText crawl-300d-2M - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix-crawl-300d.png')
    plt.close()

    print("Process completed successfully!")
    print(f"Results saved in: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()