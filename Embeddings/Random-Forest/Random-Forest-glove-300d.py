import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs('Sarcasm_outputs', exist_ok=True)

def load_glove_pkl(glove_path):
    """Load GloVe embeddings from pickle file"""
    return joblib.load(glove_path)

def text_to_vector(text, embeddings):
    """Convert text to average vector using GloVe embeddings"""
    words = text.split()
    vectors = [embeddings.get(word, np.zeros(300)) for word in words]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

def preprocess_data(df):
    """Clean and prepare the comments"""
    df = df.dropna(subset=['comment'])
    df['comment'] = df['comment'].str.lower()
    df['comment'] = df['comment'].str.replace('[^\w\s]', '', regex=True)
    return df.sample(100000, random_state=42)  # Subsample for efficiency

def main():
    # Load and preprocess data
    df = pd.read_csv('../../../train-balanced-sarcasm.csv', usecols=['label', 'comment'])
    df = preprocess_data(df)
    
    # Load 300D GloVe embeddings
    glove_embeddings = load_glove_pkl('../../../glove.840B.300d.pkl')
    
    # Convert comments to vectors
    X = np.array([text_to_vector(text, glove_embeddings) for text in df['comment']])
    y = df['label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = rf.predict(X_test)
    
    # Save classification report
    report = classification_report(y_test, y_pred)
    with open('Sarcasm_outputs/classification_report-glove-300d.txt', 'w') as f:
        f.write(report)
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarc', 'Sarc'],
                yticklabels=['Non-Sarc', 'Sarc'])
    plt.title('GloVe 840B.300d - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Sarcasm_outputs/confusion_matrix-glove-300d.png')
    plt.close()

    print("Processing complete! Results saved in Sarcasm_outputs/")

if __name__ == '__main__':
    main()