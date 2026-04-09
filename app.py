from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import FakeNewsPredictor
import os

app = Flask(__name__, static_folder='public')
CORS(app)

# Initialize the predictor
predictor = FakeNewsPredictor()

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data.get('text', '')
    
    if not news_text:
        return jsonify({'error': 'No text provided'}), 400
        
    try:
        if not predictor.is_model_loaded():
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
            
        result = predictor.predict(news_text)
        return jsonify({
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.is_model_loaded()
    })

# Serve static files for local testing
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('public', path)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
