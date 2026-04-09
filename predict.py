import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk packages are available silently
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

class FakeNewsPredictor:
    def __init__(self, model_path="models/best_model.pkl", vectorizer_path="models/tfidf_vectorizer.pkl"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self._load_models()

    def _load_models(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            return False

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        return True

    def is_model_loaded(self):
        return self.model is not None and self.vectorizer is not None

    def predict(self, text):
        if not self.is_model_loaded():
            raise Exception("Model or vectorizer not found. Please run train_model.py first.")
            
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
            
        cleaned_text = preprocess_text(text)
        
        if not cleaned_text:
            raise ValueError("Text contains no useful words after preprocessing.")
            
        vectorized_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)[0]
        
        # Calculate confidence score if the model supports predict_proba
        try:
            probabilities = self.model.predict_proba(vectorized_text)[0]
            confidence = max(probabilities) * 100
        except AttributeError:
            # For models like PassiveAggressiveClassifier, predict_proba is not available by default
            decision_val = self.model.decision_function(vectorized_text)[0]
            # Convert decision value to a somewhat representative confidence score (sigmoid)
            import numpy as np
            confidence = (1 / (1 + np.exp(-np.abs(decision_val)))) * 100

        return {
            'prediction': prediction,
            'confidence': confidence
        }

if __name__ == "__main__":
    # Test script locally
    predictor = FakeNewsPredictor()
    if predictor.is_model_loaded():
        test_text = "This is a breaking news report from a trusted source."
        print(f"Testing with text: '{test_text}'")
        print(predictor.predict(test_text))
    else:
        print("Model not found. Run train_model.py first.")
