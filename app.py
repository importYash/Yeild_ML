"""Main Flask application for yield prediction."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

from gee.gee_initializer import initialize_gee, check_gee_status
from gee.extract_features import extract_all_features

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Google Earth Engine on startup
try:
    initialize_gee()
except Exception as e:
    print(f"Warning: Failed to initialize GEE: {str(e)}")

@app.route('/')
def home():
    """Home endpoint."""
    return jsonify({
        'message': 'Yield Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'gee_initialized': check_gee_status()
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    gee_status = check_gee_status()
    return jsonify({
        'status': 'healthy' if gee_status else 'degraded',
        'gee_initialized': gee_status
    }), 200 if gee_status else 503

@app.route('/api/extract-features', methods=['POST'])
def extract_features():
    """
    Extract features for a given location.
    
    Expected JSON body:
    {
        "lat": float,
        "lon": float,
        "radius_m": int (optional, default 5000),
        "start_date": "YYYY-MM-DD" (optional),
        "end_date": "YYYY-MM-DD" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        if 'lat' not in data or 'lon' not in data:
            return jsonify({'error': 'lat and lon are required'}), 400
        
        lat = float(data['lat'])
        lon = float(data['lon'])
        radius_m = int(data.get('radius_m', 5000))
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Extract features
        features = extract_all_features(lat, lon, radius_m, start_date, end_date)
        
        if features is None:
            return jsonify({'error': 'Failed to extract features'}), 500
        
        return jsonify({
            'success': True,
            'features': features
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/predict-yield', methods=['POST'])
def predict_yield():
    """
    Predict yield for a given location.
    
    Expected JSON body:
    {
        "lat": float,
        "lon": float,
        "crop_type": string (optional),
        "radius_m": int (optional, default 2000)
    }
    """
    try:
        from utils.ml_model import get_predictor
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        if 'lat' not in data or 'lon' not in data:
            return jsonify({'error': 'lat and lon are required'}), 400
        
        lat = float(data['lat'])
        lon = float(data['lon'])
        crop_type = data.get('crop_type', 'oilseed')
        radius_m = int(data.get('radius_m', 2000))
        
        # Extract features
        features = extract_all_features(lat, lon, radius_m)
        
        if features is None:
            return jsonify({'error': 'Failed to extract features'}), 500
        
        # Get ML model predictor
        predictor = get_predictor()
        
        # Make prediction
        prediction = predictor.predict(features)
        
        # Check if there's a validation error
        if prediction.get('error'):
            return jsonify({
                'success': True,
                'prediction': prediction,
                'features': features
            })
        
        return jsonify({
            'success': True,
            'prediction': {
                'crop_type': crop_type,
                'predicted_yield': prediction['predicted_yield'],
                'unit': 'tons/hectare',
                'confidence': prediction['confidence'],
                'model_used': prediction.get('model_used', 'Unknown'),
                'model_r2': prediction.get('model_r2'),
                'conditions': prediction.get('conditions'),
                'benchmarks': prediction.get('benchmarks')
            },
            'features': features
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Yield Prediction API Server")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
