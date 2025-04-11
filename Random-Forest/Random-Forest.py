import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Create directories if they don't exist
os.makedirs('results/random_forest', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load dataset
def load_data(file_path, n_samples=100000):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    return df.sample(n_samples, random_state=42)  # Sample for faster execution

# Preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.replace('\n', ' ').replace('\r', '')  # Remove newlines
    text = ''.join([c for c in text if c.isalnum() or c in ' '])  # Remove special chars
    return text.strip()

# Main function
def main():
    # Load and prepare data
    df = load_data('../../train-balanced-sarcasm.csv')
    df['cleaned_comment'] = df['comment'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_comment'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, 
                                n_jobs=-1, 
                                class_weight='balanced',
                                random_state=42)
    rf.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test_tfidf)
    
    # Generate reports
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save reports
    with open('results/random_forest/classification_report.txt', 'w') as f:
        f.write(report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('results/random_forest/confusion_matrix.png')
    plt.close()
    
    # Save model and vectorizer
    joblib.dump(rf, 'models/random_forest_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    
    print("Training completed! Results saved in results/random_forest/")

if __name__ == '__main__':
    main()