import os
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download necessary NLTK datasets
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove punctuation & special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stopwords, then lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def train_models():
    data_path = os.path.join("data", "dataset.csv")
    if not os.path.exists(data_path):
        print("Dataset not found. Please run data_downloader.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # We will use the 'text' column for training. Sometimes combining title and text is better.
    print("Preprocessing text... This might take a minute or two.")
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    # Combining title and text for better context
    df['content'] = df['title'] + " " + df['text']
    
    # Sample the data if it's too large to train quickly, but since fake_or_real_news is ~6.3k rows, we can use it all
    selected_data = df[['content', 'label']].copy()
    selected_data['clean_content'] = selected_data['content'].apply(preprocess_text)
    
    # Features and Targets
    X = selected_data['clean_content']
    y = selected_data['label']
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Extracting features with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    
    # Train Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Passive Aggressive Classifier': PassiveAggressiveClassifier(max_iter=50)
    }
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    print("\nTraining and Evaluating Models:")
    for name, model in models.items():
        print(f"--> Training {name}...")
        model.fit(tfidf_train, y_train)
        pred = model.predict(tfidf_test)
        acc = accuracy_score(y_test, pred)
        print(f"Accuracy for {name}: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with Accuracy: {best_acc*100:.2f}%")
    print("\nConfusion Matrix for Best Model:")
    print(confusion_matrix(y_test, best_model.predict(tfidf_test)))
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(tfidf_test)))
    
    # Save the best model and vectorizer
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "best_model.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
        
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    train_models()
