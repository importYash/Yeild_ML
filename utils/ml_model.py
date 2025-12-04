"""
ML Model utilities for oilseed yield prediction
"""
import joblib
import os
import numpy as np
from pathlib import Path

class OilseedYieldPredictor:
    """Wrapper class for the trained oilseed yield prediction model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.stats = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained oilseed model and scaler from disk"""
        try:
            model_dir = Path(__file__).parent.parent.parent / 'models'
            model_path = model_dir / 'oilseed_yield_model.pkl'
            scaler_path = model_dir / 'feature_scaler.pkl'
            stats_path = model_dir / 'oilseed_model_stats.pkl'
            
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                if stats_path.exists():
                    self.stats = joblib.load(stats_path)
                self.model_loaded = True
                print("[OK] Oilseed yield prediction model loaded successfully")
                if self.stats:
                    print(f"    Model Type: {self.stats.get('model_type', 'Unknown')}")
                    print(f"    Test R²: {self.stats['test_r2']:.4f}")
            else:
                print("[WARNING] Model files not found - using fallback")
                
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            self.model_loaded = False
    
    def predict(self, features):
        """Predict yield from features"""
        print(f"\n[PREDICT] Model loaded: {self.model_loaded}")
        
        if not self.model_loaded:
            return self._fallback_prediction(features)
        
        try:
            ndvi = features.get('ndvi', {})
            weather = features.get('weather', {})
            
            mean_ndvi = ndvi.get('mean_ndvi', 0.5)
            min_ndvi = ndvi.get('min_ndvi', 0.0)
            max_ndvi = ndvi.get('max_ndvi', 1.0)
            image_count = ndvi.get('image_count', 20)
            mean_temp = weather.get('mean_temperature', 22)
            total_precip = weather.get('total_precipitation', 0.6)
            
            # Validation
            validation = self._validate_agricultural_area(mean_ndvi, min_ndvi, max_ndvi)
            if not validation['is_agricultural']:
                return {
                    'error': True,
                    'error_type': validation['area_type'],
                    'message': validation['message'],
                    'predicted_yield': 0.0,
                    'confidence': 0.0,
                    'model_used': 'Validation Check',
                    'conditions': validation['area_type'],
                    'ndvi_info': {'mean': mean_ndvi, 'min': min_ndvi, 'max': max_ndvi}
                }
            
            print(f"[DEBUG] NDVI: {mean_ndvi:.4f}, Temp: {mean_temp:.2f}°C")
            
            # Clip values
            mean_ndvi = np.clip(mean_ndvi, 0.2, 0.95)
            mean_temp = np.clip(mean_temp, 5, 45)
            total_precip = np.clip(total_precip, 0.1, 2.0)
            image_count = np.clip(image_count, 1, 50)
            
            # Predict
            X = np.array([[mean_ndvi, mean_temp, total_precip, image_count]])
            X_scaled = self.scaler.transform(X)
            predicted_yield = self.model.predict(X_scaled)[0]
            predicted_yield = np.clip(predicted_yield, 0.5, 5.0)
            
            print(f"[DEBUG] Prediction: {predicted_yield:.2f} tons/ha")
            
            confidence = self._calculate_confidence(mean_ndvi, mean_temp, total_precip, image_count)
            
            return {
                'predicted_yield': float(round(predicted_yield, 2)),
                'confidence': float(round(confidence, 3)),
                'model_used': 'Gradient Boosting (Oilseed-Specific)',
                'model_accuracy': self.stats['test_r2'] if self.stats else None,
                'conditions': self._assess_conditions(mean_ndvi, mean_temp, total_precip),
                'benchmarks': self._get_global_benchmarks(predicted_yield)
            }
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._fallback_prediction(features)
    
    def _validate_agricultural_area(self, mean_ndvi, min_ndvi, max_ndvi):
        """Validate if area is agricultural"""
        if mean_ndvi < 0.1 and max_ndvi < 0.2:
            return {
                'is_agricultural': False,
                'area_type': 'Non-Agricultural (Water Bodies)',
                'message': f'Water bodies detected (NDVI: {mean_ndvi:.3f}). Please select an agricultural field.'
            }
        
        if mean_ndvi < 0.15 and max_ndvi < 0.3:
            return {
                'is_agricultural': False,
                'area_type': 'Non-Agricultural (Buildings/Urban)',
                'message': f'Built-up area detected (NDVI: {mean_ndvi:.3f}). Please select an agricultural field.'
            }
        
        if max_ndvi >= 0.4:
            return {
                'is_agricultural': True,
                'area_type': 'Agricultural Land',
                'message': f'Valid agricultural area (Mean: {mean_ndvi:.3f}, Max: {max_ndvi:.3f})'
            }
        
        if mean_ndvi < 0.2 and max_ndvi < 0.35:
            return {
                'is_agricultural': False,
                'area_type': 'Bare Soil/No Vegetation',
                'message': f'No healthy vegetation (NDVI: {mean_ndvi:.3f}). Please select a field with crops.'
            }
        
        return {
            'is_agricultural': True,
            'area_type': 'Agricultural Land',
            'message': 'Valid agricultural area detected'
        }
    
    def _get_global_benchmarks(self, predicted_yield):
        """Get global benchmarking data"""
        benchmarks = {
            'world_average': 2.1,
            'top_producers': {
                'Canada': 2.8,
                'Australia': 2.5,
                'India': 1.4,
                'China': 2.3,
                'France': 3.2
            }
        }
        
        vs_world = ((predicted_yield - benchmarks['world_average']) / benchmarks['world_average']) * 100
        
        if predicted_yield >= 3.0:
            rating, rating_color = 'Excellent', 'success'
        elif predicted_yield >= 2.5:
            rating, rating_color = 'Good', 'primary'
        elif predicted_yield >= 2.0:
            rating, rating_color = 'Average', 'warning'
        else:
            rating, rating_color = 'Below Average', 'error'
        
        return {
            'world_average': benchmarks['world_average'],
            'vs_world_percent': round(vs_world, 1),
            'rating': rating,
            'rating_color': rating_color,
            'top_producers': benchmarks['top_producers'],
            'percentile': self._calculate_percentile(predicted_yield)
        }
    
    def _calculate_percentile(self, yield_value):
        """Calculate global percentile"""
        if yield_value >= 3.2:
            return 95
        elif yield_value >= 2.8:
            return 85
        elif yield_value >= 2.5:
            return 75
        elif yield_value >= 2.1:
            return 50
        elif yield_value >= 1.8:
            return 35
        elif yield_value >= 1.4:
            return 20
        else:
            return 10
    
    def _calculate_confidence(self, ndvi, temp, precip, image_count):
        """Calculate prediction confidence"""
        ndvi_conf = 1.0 if 0.4 <= ndvi <= 0.8 else 0.7
        temp_conf = 1.0 if 20 <= temp <= 25 else 0.8
        precip_conf = 1.0 if 0.5 <= precip <= 0.7 else 0.8
        image_conf = 1.0 if image_count >= 20 else 0.8
        
        confidence = (ndvi_conf * 0.3 + temp_conf * 0.25 + precip_conf * 0.35 + image_conf * 0.1)
        
        if self.stats:
            confidence = confidence * (0.7 + 0.3 * self.stats['test_r2'])
        
        return max(0.6, min(0.98, confidence))
    
    def _assess_conditions(self, ndvi, temp, precip):
        """Assess growing conditions"""
        if 0.6 <= ndvi <= 0.8 and 20 <= temp <= 25 and 0.5 <= precip <= 0.7:
            return "Excellent"
        elif 0.5 <= ndvi <= 0.75 and 18 <= temp <= 28 and 0.4 <= precip <= 0.8:
            return "Good"
        elif 0.4 <= ndvi <= 0.6:
            return "Fair"
        else:
            return "Stressed"
    
    def _fallback_prediction(self, features):
        """Fallback when model not available"""
        ndvi = features.get('ndvi', {})
        weather = features.get('weather', {})
        
        mean_ndvi = np.clip(ndvi.get('mean_ndvi', 0.5), 0.3, 0.85)
        mean_temp = np.clip(weather.get('mean_temperature', 22), 12, 35)
        total_precip = np.clip(weather.get('total_precipitation', 0.6), 0.25, 1.0)
        
        base_yield = 0.5 + (mean_ndvi - 0.3) * 5.5
        temp_factor = 1.0 if 20 <= mean_temp <= 25 else max(0.5, 1.0 - abs(mean_temp - 22.5) / 25)
        
        precip_mm = total_precip * 1000
        water_factor = 1.0 if 500 <= precip_mm <= 700 else max(0.5, 1.0 - abs(precip_mm - 600) / 800)
        
        predicted_yield = base_yield * temp_factor * water_factor
        predicted_yield = np.clip(predicted_yield, 0.8, 4.5)
        
        return {
            'predicted_yield': float(round(predicted_yield, 2)),
            'confidence': 0.70,
            'model_used': 'Empirical Formula (Fallback)',
            'model_accuracy': None,
            'conditions': 'Unknown',
            'benchmarks': self._get_global_benchmarks(predicted_yield)
        }

# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create the global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = OilseedYieldPredictor()
    return _predictor
