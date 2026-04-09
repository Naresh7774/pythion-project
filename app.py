from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import FakeNewsPredictor
import os
import traceback

app = Flask(__name__, static_folder='public')
CORS(app)

# Initialize the predictor
try:
    predictor = FakeNewsPredictor()
except Exception as e:
    print(f"CRITICAL: Failed to initialize predictor: {str(e)}")
    predictor = None

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON request'}), 400
            
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({'error': 'No text provided for analysis'}), 400
            
        if not predictor or not predictor.is_model_loaded():
            # Try to re-initialize or return error
            return jsonify({
                'error': 'Machine learning model is not loaded on the server.',
                'details': 'Ensure models are trained and pushed to GitHub.'
            }), 500
            
        result = predictor.predict(news_text)
        return jsonify({
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
    except Exception as e:
        # Return full error to frontend for debugging
        error_details = traceback.format_exc()
        print(f"ERROR during prediction: {error_details}")
        return jsonify({
            'error': str(e),
            'details': error_details if os.environ.get('VERCEL_ENV') else 'Internal Server Error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.is_model_loaded() if predictor else False
    })

# Serve static files
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('public', path)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
