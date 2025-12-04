"""Google Earth Engine initialization module."""
import ee
import os
import json
from pathlib import Path

def initialize_gee():
    """Initialize Google Earth Engine with service account credentials."""
    try:
        # Get the path to the service account key
        key_path = Path(__file__).parent.parent / 'keys' / 'gee-private-key.json'
        
        if not key_path.exists():
            raise FileNotFoundError(f"Service account key not found at {key_path}")
        
        # Load service account credentials
        with open(key_path, 'r') as f:
            credentials_info = json.load(f)
        
        service_account = credentials_info['client_email']
        
        # Initialize Earth Engine with service account
        credentials = ee.ServiceAccountCredentials(service_account, str(key_path))
        ee.Initialize(credentials)
        
        print("[OK] Google Earth Engine initialized successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error initializing Google Earth Engine: {str(e)}")
        raise

def check_gee_status():
    """Check if GEE is properly initialized."""
    try:
        # Try a simple operation to verify initialization
        ee.Number(1).getInfo()
        return True
    except Exception as e:
        print(f"GEE not initialized: {str(e)}")
        return False
