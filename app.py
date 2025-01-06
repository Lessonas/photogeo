from flask import Flask, request, render_template, jsonify, current_app
from flask_caching import Cache
from PIL import Image
import exifread
from geopy.geocoders import Nominatim, GoogleV3
from geopy.exc import GeocoderTimedOut
import os
from datetime import datetime
import folium
import requests
import re
import io
import json
import tempfile
import urllib.parse
from google.cloud import vision
import numpy as np
from dotenv import load_dotenv
import redis
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(handler)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-123')

# Configure caching
cache_config = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    'CACHE_DEFAULT_TIMEOUT': 3600,
    'CACHE_KEY_PREFIX': 'photogeo_'
}
app.config.update(cache_config)
cache = Cache(app)

# Initialize services with error handling
try:
    # Initialize Google Cloud Vision client with credentials from environment
    credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
    if not credentials_json:
        raise ValueError("GOOGLE_CREDENTIALS_JSON environment variable not set")
        
    # Create credentials from JSON string
    import json
    from google.oauth2 import service_account
    import tempfile
    
    # Write credentials to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(json.loads(credentials_json), temp_file)
        temp_credentials_path = temp_file.name
    
    # Set the environment variable to point to our temporary file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials_path
    
    # Initialize vision client
    vision_client = vision.ImageAnnotatorClient()
    
    # Clean up the temporary file
    os.unlink(temp_credentials_path)
    
    # Initialize Redis client
    redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    executor = ThreadPoolExecutor(max_workers=3)
    logger.info('Successfully initialized all services')
except Exception as e:
    logger.error(f'Error initializing services: {str(e)}')
    raise

def get_location_with_retry(geolocator, query, max_retries=3):
    """Retry geocoding with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return geolocator.geocode(query, exactly_one=True)
        except GeocoderTimedOut:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

@cache.memoize(timeout=86400)  # Cache for 24 hours
def get_location_name(lat, lon):
    """Enhanced reverse geocoding with multiple providers and caching."""
    try:
        # Try Nominatim first
        geolocator = Nominatim(user_agent="photo_geolocator")
        location = get_location_with_retry(geolocator, f"{lat}, {lon}")
        
        if location and location.address:
            return location.address
            
        # Fallback to alternative geocoding if needed
        return f"Location at {lat:.6f}, {lon:.6f}"
    except Exception as e:
        logger.error(f'Geocoding error: {str(e)}')
        return f"Location at {lat:.6f}, {lon:.6f}"

def create_map(lat, lon, zoom=15):
    """Create an enhanced interactive map with multiple layers."""
    try:
        m = folium.Map(location=[lat, lon], zoom_start=zoom)
        
        # Add multiple tile layers
        folium.TileLayer('openstreetmap').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite'
        ).add_to(m)
        
        # Add marker with popup
        location_name = get_location_name(lat, lon)
        popup_html = f"""
        <div style="width:200px">
            <h4>{location_name}</h4>
            <p>Coordinates: {lat:.6f}, {lon:.6f}</p>
            <p>
                <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank">Open in Google Maps</a><br>
                <a href="https://www.openstreetmap.org/?mlat={lat}&mlon={lon}" target="_blank">Open in OpenStreetMap</a>
            </p>
        </div>
        """
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Add circle for accuracy visualization
        folium.Circle(
            radius=100,
            location=[lat, lon],
            color="red",
            fill=True,
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    except Exception as e:
        logger.error(f'Map creation error: {str(e)}')
        raise

@cache.memoize(timeout=3600)
def detect_location_google_vision(image_path):
    """Enhanced location detection using Google Cloud Vision API."""
    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        
        # Run multiple detection types in parallel
        landmark_future = executor.submit(vision_client.landmark_detection, image)
        label_future = executor.submit(vision_client.label_detection, image)
        text_future = executor.submit(vision_client.text_detection, image)
        
        # Gather results
        landmark_response = landmark_future.result()
        label_response = label_future.result()
        text_response = text_future.result()
        
        results = {
            'locations': [],
            'labels': [],
            'text': []
        }
        
        # Process landmark detection
        if landmark_response.landmark_annotations:
            for landmark in landmark_response.landmark_annotations:
                if landmark.locations:
                    lat_lng = landmark.locations[0].lat_lng
                    results['locations'].append({
                        'name': landmark.description,
                        'lat': lat_lng.latitude,
                        'lng': lat_lng.longitude,
                        'confidence': landmark.score,
                        'source': 'landmark'
                    })
        
        # Process labels for context
        if label_response.label_annotations:
            results['labels'] = [{
                'description': label.description,
                'confidence': label.score
            } for label in label_response.label_annotations]
        
        # Process text for location hints
        if text_response.text_annotations:
            results['text'] = text_response.text_annotations[0].description if text_response.text_annotations else ''
            
        return results
    except Exception as e:
        logger.error(f'Google Vision API error: {str(e)}')
        return None

def analyze_image_context(predictions):
    """Enhanced context analysis with multiple data points."""
    try:
        if not predictions:
            return None
            
        # Check landmark detection results
        if 'locations' in predictions and predictions['locations']:
            # Sort by confidence and return the best match
            locations = sorted(predictions['locations'], key=lambda x: x.get('confidence', 0), reverse=True)
            best_location = locations[0]
            return {
                'latitude': best_location['lat'],
                'longitude': best_location['lng'],
                'confidence': best_location['confidence'],
                'name': best_location.get('name', 'Unknown Location'),
                'source': best_location.get('source', 'vision_api')
            }
            
        # If no direct location found, try to infer from labels and text
        location_hints = []
        
        # Check labels for location hints
        if 'labels' in predictions:
            location_hints.extend([label['description'] for label in predictions['labels']])
            
        # Check text for location information
        if 'text' in predictions and predictions['text']:
            extracted_locations = extract_location_from_text(predictions['text'])
            if extracted_locations:
                location_hints.extend(extracted_locations)
        
        # Try to geocode location hints
        if location_hints:
            geolocator = Nominatim(user_agent="photo_geolocator")
            for hint in location_hints:
                try:
                    location = get_location_with_retry(geolocator, hint)
                    if location:
                        return {
                            'latitude': location.latitude,
                            'longitude': location.longitude,
                            'confidence': 0.5,  # Lower confidence for inferred locations
                            'name': location.address,
                            'source': 'inference'
                        }
                except Exception as e:
                    logger.warning(f'Geocoding error for hint {hint}: {str(e)}')
                    continue
                    
        return None
    except Exception as e:
        logger.error(f'Context analysis error: {str(e)}')
        return None

def extract_location_from_text(text):
    """Extract potential location information from text using various patterns."""
    # Common patterns for coordinates
    coord_patterns = [
        r'(\d+°\d+\'(?:\d+(?:\.\d+)?)?\"[NS])[,\s]+(\d+°\d+\'(?:\d+(?:\.\d+)?)?\"[EW])',  # DMS format
        r'(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)',  # Decimal format
        r'@(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)',  # Social media format
        r'location[:/].*?(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)',  # Location prefix format
    ]
    
    for pattern in coord_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    
    return None

def download_image(url):
    """Download image from URL and save to temporary file."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp.write(response.content)
        temp.close()
        
        return temp.name
    except Exception as e:
        logger.error(f'Error downloading image: {str(e)}')
        return None

def get_exif_data(image_path):
    """Extract EXIF data from image."""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
        exif_data = {
            'gps_latitude': None,
            'gps_longitude': None,
            'datetime': None,
            'make': str(tags.get('Image Make', '')),
            'model': str(tags.get('Image Model', '')),
        }
        
        # Get GPS data
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat = tags.get('GPS GPSLatitude')
            lat_ref = tags.get('GPS GPSLatitudeRef')
            lon = tags.get('GPS GPSLongitude')
            lon_ref = tags.get('GPS GPSLongitudeRef')
            
            if all([lat, lat_ref, lon, lon_ref]):
                lat = convert_to_degrees(lat.values)
                lon = convert_to_degrees(lon.values)
                
                if str(lat_ref) == 'S': lat = -lat
                if str(lon_ref) == 'W': lon = -lon
                
                exif_data['gps_latitude'] = lat
                exif_data['gps_longitude'] = lon
        
        # Get datetime
        if 'EXIF DateTimeOriginal' in tags:
            exif_data['datetime'] = str(tags['EXIF DateTimeOriginal'])
        
        return exif_data
    except Exception as e:
        logger.error(f'Error reading EXIF: {str(e)}')
        return None

def convert_to_degrees(values):
    """Convert GPS coordinates to degrees."""
    d = float(values[0].num) / float(values[0].den)
    m = float(values[1].num) / float(values[1].den)
    s = float(values[2].num) / float(values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

def process_image(image_path):
    """Process image and get location data."""
    try:
        # Try multiple methods to get location in parallel
        location_data = None
        futures = []
        
        with ThreadPoolExecutor() as executor:
            # Start all detection methods in parallel
            futures.append(executor.submit(get_exif_data, image_path))
            futures.append(executor.submit(detect_location_google_vision, image_path))
            futures.append(executor.submit(detect_location_ai, image_path))
            
            # Process results as they complete
            for future in futures:
                result = future.result()
                if result:
                    if isinstance(result, dict) and 'latitude' in result:
                        location_data = result
                        break
                    elif isinstance(result, list):  # AI predictions
                        context_location = analyze_image_context(result)
                        if context_location:
                            location_data = context_location
                            break

        if not location_data:
            logger.warning('No location data found in image')
            return {
                'error': 'No location data found in image',
                'processing_time': str(datetime.now() - datetime.now())
            }
        
        # Get detailed location information
        location_name = get_location_name(location_data['latitude'], location_data['longitude'])
        map_path = create_map(location_data['latitude'], location_data['longitude'])
        
        # Get nearby points of interest
        nearby = rg.search((location_data['latitude'], location_data['longitude']))
        
        return {
            'latitude': location_data['latitude'],
            'longitude': location_data['longitude'],
            'location': location_name,
            'source': location_data['source'],
            'confidence': location_data.get('confidence', 0.0),
            'nearby': nearby[0] if nearby else None,
            'detected_objects': location_data.get('detected_objects', []),
            'processing_time': str(datetime.now() - datetime.now()),
            'map': bool(map_path)
        }
    except Exception as e:
        logger.error(f'Error processing image: {str(e)}')
        return {
            'error': str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Create uploads directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        try:
            # Get location data from image
            with ThreadPoolExecutor() as executor:
                # Start all detection methods in parallel
                futures = []
                futures.append(executor.submit(get_exif_data, filepath))
                futures.append(executor.submit(detect_location_google_vision, filepath))
                
                # Process results as they complete
                location_data = None
                for future in futures:
                    try:
                        result = future.result()
                        if result and result.get('confidence', 0) > (location_data.get('confidence', 0) if location_data else 0):
                            location_data = result
                    except Exception as e:
                        logger.error(f'Error in detection method: {str(e)}')

            if not location_data:
                return jsonify({
                    'error': 'No location data found in image'
                }), 404

            # Get detailed location information
            location_name = get_location_name(location_data['latitude'], location_data['longitude'])
            
            # Get nearby points of interest
            nearby = rg.search((location_data['latitude'], location_data['longitude']))

            response_data = {
                'latitude': location_data['latitude'],
                'longitude': location_data['longitude'],
                'location': location_name,
                'confidence': location_data.get('confidence', 0.0),
                'nearby': nearby[0] if nearby else None,
                'detected_objects': location_data.get('detected_objects', [])
            }

            return jsonify(response_data), 200

        except Exception as e:
            logger.error(f'Error processing image: {str(e)}')
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up the uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f'Error removing temporary file: {str(e)}')

    except Exception as e:
        logger.error(f'Error handling upload: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
