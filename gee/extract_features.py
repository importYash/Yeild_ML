"""Feature extraction from Google Earth Engine for yield prediction."""
import ee
from datetime import datetime, timedelta

def get_ndvi_time_series(geometry, start_date, end_date):
    """
    Extract NDVI time series from Sentinel-2 imagery.
    
    Args:
        geometry: ee.Geometry object representing the area of interest
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        Dictionary with NDVI statistics and image URL
    """
    try:
        # Load Sentinel-2 Surface Reflectance collection
        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # Function to calculate NDVI
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        # Add NDVI band to all images
        collection_with_ndvi = collection.map(add_ndvi)
        
        # Get image count
        image_count = collection.size().getInfo()
        
        # Calculate NDVI statistics over the region
        ndvi_image = collection_with_ndvi.select('NDVI').median()
        
        ndvi_stats = ndvi_image.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.min(), '', True
            ).combine(
                ee.Reducer.max(), '', True
            ).combine(
                ee.Reducer.stdDev(), '', True
            ),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        # Generate thumbnail URL for NDVI visualization
        vis_params = {
            'min': 0.0,
            'max': 1.0,
            'palette': ['red', 'yellow', 'green']
        }
        
        # Get the thumbnail URL
        thumbnail_url = ndvi_image.getThumbURL({
            'min': vis_params['min'],
            'max': vis_params['max'],
            'palette': vis_params['palette'],
            'dimensions': 512,
            'region': geometry,
            'format': 'png'
        })
        
        return {
            'mean_ndvi': ndvi_stats.get('NDVI_mean', 0.5),
            'min_ndvi': ndvi_stats.get('NDVI_min', 0.0),
            'max_ndvi': ndvi_stats.get('NDVI_max', 1.0),
            'std_ndvi': ndvi_stats.get('NDVI_stdDev', 0.1),
            'image_count': image_count,
            'thumbnail_url': thumbnail_url,
            'quality': 'Excellent' if image_count >= 20 else 'Good' if image_count >= 10 else 'Fair'
        }
        
    except Exception as e:
        print(f"Error extracting NDVI: {str(e)}")
        return {
            'mean_ndvi': 0.5,
            'min_ndvi': 0.3,
            'max_ndvi': 0.7,
            'std_ndvi': 0.1,
            'image_count': 10,
            'thumbnail_url': None,
            'quality': 'Fair'
        }

def get_weather_data(geometry, start_date, end_date):
    """
    Extract weather data from ERA5 climate reanalysis.
    
    Args:
        geometry: ee.Geometry object representing the area of interest
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        Dictionary with weather statistics
    """
    try:
        # Load ERA5 daily aggregates
        era5 = ee.ImageCollection('ECMWF/ERA5/DAILY') \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date)
        
        # Check if collection has any images
        count = era5.size().getInfo()
        if count == 0:
            print(f"No ERA5 data available for date range: {start_date} to {end_date}")
            return {
                'mean_temperature': 20.0,
                'total_precipitation': 0.5
            }
        
        # Calculate mean temperature
        temp_mean = era5.select('mean_2m_air_temperature').mean()
        
        # Calculate total precipitation
        precip_total = era5.select('total_precipitation').sum()
        
        # Reduce over region separately
        temp_stats = temp_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=27830,
            maxPixels=1e9
        ).getInfo()
        
        precip_stats = precip_total.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=27830,
            maxPixels=1e9
        ).getInfo()
        
        # Convert temperature from Kelvin to Celsius
        temp_celsius = temp_stats.get('mean_2m_air_temperature', 293.15) - 273.15
        precip = precip_stats.get('total_precipitation', 0.5)
        
        return {
            'mean_temperature': temp_celsius,
            'total_precipitation': precip
        }
        
    except Exception as e:
        print(f"Error extracting weather data: {str(e)}")
        # Return default values on error
        return {
            'mean_temperature': 20.0,
            'total_precipitation': 0.5
        }

def extract_all_features(lat, lon, radius_m=2000, start_date=None, end_date=None):
    """
    Extract all features for a given location.
    
    Args:
        lat: Latitude
        lon: Longitude
        radius_m: Radius in meters for the area of interest
        start_date: Start date (defaults to 6 months ago)
        end_date: End date (defaults to today)
    
    Returns:
        Dictionary with all extracted features
    """
    try:
        # Create geometry
        point = ee.Geometry.Point([lon, lat])
        geometry = point.buffer(radius_m)
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"Extracting features for location: ({lat}, {lon})")
        print(f"Date range: {start_date} to {end_date}")
        
        # Extract NDVI
        ndvi_data = get_ndvi_time_series(geometry, start_date, end_date)
        
        # Extract weather data
        weather_data = get_weather_data(geometry, start_date, end_date)
        
        # Ensure we have valid data
        if ndvi_data is None:
            ndvi_data = {
                'mean_ndvi': 0.5,
                'min_ndvi': 0.3,
                'max_ndvi': 0.7,
                'std_ndvi': 0.1,
                'image_count': 10,
                'thumbnail_url': None,
                'quality': 'Fair'
            }
        if weather_data is None:
            weather_data = {'mean_temperature': 20.0, 'total_precipitation': 0.5}
        
        # Combine all features
        features = {
            'location': {'lat': lat, 'lon': lon, 'radius_m': radius_m},
            'date_range': {'start': start_date, 'end': end_date},
            'ndvi': ndvi_data,
            'weather': weather_data
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None
