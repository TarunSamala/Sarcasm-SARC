import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Create directories
os.makedirs('results/random_forest_glove', exist_ok=True)
os.makedirs('models/glove', exist_ok=True)

def load_glove(glove_path):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def text_to_vector(text, embeddings):
    words = text.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

def load_data(file_path, n_samples=100000):
    df = pd.read_csv(file_path, usecols=['label', 'comment'])
    df = df.dropna(subset=['comment'])
    df = df.sample(n_samples, random_state=42)
    df['comment'] = df['comment'].str.lower()
    df['comment'] = df['comment'].str.replace('[^\w\s]', '', regex=True)
    return df

def main():
    # Load and prepare data
    df = load_data('../../../train-balanced-sarcasm.csv')
    
    # Load GloVe embeddings
    glove_embeddings = load_glove('../../../glove.6B.100d.txt')
    
    # Convert text to vectors
    X = np.array([text_to_vector(text, glove_embeddings) for text in df['comment']])
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Generate reports
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save reports
    with open('results/random_forest_glove/classification_report.txt', 'w') as f:
        f.write(report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (GloVe Embeddings)')
    plt.savefig('results/random_forest_glove/confusion_matrix.png')
    plt.close()
    
    # Save model
    joblib.dump(rf, 'models/glove/random_forest_glove_model.pkl')
    
    print("GloVe-based Random Forest training completed!")
    print("Results saved in results/random_forest_glove/")

if __name__ == '__main__':
    main()