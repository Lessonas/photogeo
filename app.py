from flask import Flask, request, render_template, jsonify
from flask_caching import Cache
from PIL import Image
import exifread
from geopy.geocoders import Nominatim
import os
from datetime import datetime
import folium
import requests
import re
import io
import json
import tempfile
import urllib.parse
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from google.cloud import vision
import reverse_geocoder as rg
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from dotenv import load_dotenv
from mapbox import Geocoder
import redis
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(handler)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-this-in-production')

# Configure caching
cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Initialize thread pool
executor = ThreadPoolExecutor(max_workers=3)

# Initialize AI models
try:
    # Load landmark detection model
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # Load TensorFlow model for scene recognition
    scene_model = hub.load('https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5')
    
    # Initialize Google Cloud Vision client
    vision_client = vision.ImageAnnotatorClient()
    
    # Initialize Mapbox
    geocoder = Geocoder(access_token=os.environ.get('MAPBOX_TOKEN'))
    
    logger.info('Successfully initialized all AI models and services')
except Exception as e:
    logger.error(f'Error initializing AI models: {str(e)}')

@cache.memoize(timeout=3600)
def get_location_name(lat, lon):
    """Get location name from coordinates using multiple services."""
    try:
        # Try Mapbox first
        response = geocoder.reverse(lon=lon, lat=lat)
        if response.ok:
            feature = response.geojson()['features'][0]
            return feature['place_name']
    except Exception as e:
        logger.warning(f'Mapbox geocoding failed: {str(e)}')
    
    try:
        # Fallback to Nominatim
        geolocator = Nominatim(user_agent="photo_geolocator")
        location = geolocator.reverse(f"{lat}, {lon}", language='en')
        return location.address if location else "Location not found"
    except Exception as e:
        logger.warning(f'Nominatim geocoding failed: {str(e)}')
        return "Location lookup failed"

def create_map(lat, lon, zoom=15):
    """Create an enhanced interactive map."""
    try:
        # Create Folium map with satellite and street layers
        m = folium.Map(
            location=[lat, lon],
            zoom_start=zoom,
            prefer_canvas=True,
            control_scale=True
        )
        
        # Add tile layers
        folium.TileLayer('openstreetmap').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite'
        ).add_to(m)
        
        # Add marker with popup
        folium.Marker(
            [lat, lon],
            popup='Location',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Add circle for accuracy radius
        folium.Circle(
            radius=100,
            location=[lat, lon],
            popup='Approximate Area',
            color='red',
            fill=True
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen option
        folium.plugins.Fullscreen().add_to(m)
        
        # Add custom HTML for different map views
        street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        apple_maps_url = f"https://maps.apple.com/?ll={lat},{lon}&z=15"
        bing_maps_url = f"https://www.bing.com/maps?cp={lat}~{lon}&lvl=15"
        
        html = f"""
        <div class="map-links" style="text-align: center; padding: 10px;">
            <a href="{street_view_url}" target="_blank" class="btn btn-primary" style="margin: 5px;">Street View</a>
            <a href="{apple_maps_url}" target="_blank" class="btn btn-primary" style="margin: 5px;">Apple Maps</a>
            <a href="{bing_maps_url}" target="_blank" class="btn btn-primary" style="margin: 5px;">Bing Maps</a>
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(html))
        map_path = os.path.join('static', 'map.html')
        m.save(map_path)
        return map_path
    except Exception as e:
        logger.error(f'Error creating map: {str(e)}')
        return None

def detect_location_google_vision(image_path):
    """Detect location using Google Cloud Vision API."""
    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        
        # Detect landmarks
        response = vision_client.landmark_detection(image=image)
        landmarks = response.landmark_annotations

        if landmarks:
            landmark = landmarks[0]
            lat = landmark.locations[0].lat_lng.latitude
            lon = landmark.locations[0].lat_lng.longitude
            return {
                'latitude': lat,
                'longitude': lon,
                'confidence': landmark.score,
                'name': landmark.description,
                'source': 'Google Vision API'
            }

        # Try object detection if no landmarks found
        objects = vision_client.object_localization(image=image).localized_object_annotations
        if objects:
            # Use reverse geocoding for detected objects' context
            context = [obj.name for obj in objects]
            return {
                'detected_objects': context,
                'source': 'Object Detection'
            }

        return None
    except Exception as e:
        logger.error(f'Google Vision API error: {str(e)}')
        return None

def detect_location_ai(image_path):
    """Detect location using AI model."""
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top predictions
        top_prob, top_label = torch.topk(probabilities, k=5)
        predictions = []
        for prob, label in zip(top_prob[0], top_label[0]):
            predictions.append({
                'label': model.config.id2label[label.item()],
                'confidence': prob.item()
            })
        
        return predictions
    except Exception as e:
        logger.error(f'AI model error: {str(e)}')
        return None

def analyze_image_context(predictions):
    """Analyze image context from AI predictions to estimate location."""
    try:
        # Use a geocoding service to get location from predicted labels
        geolocator = Nominatim(user_agent="photo_geolocator")
        
        for pred in predictions:
            location = geolocator.geocode(pred['label'], exactly_one=True)
            if location:
                return {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'confidence': pred['confidence'],
                    'source': 'AI Context Analysis',
                    'detected_label': pred['label']
                }
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        start_time = datetime.now()
        logger.info('Starting new upload request')
        
        file = None
        image_url = request.form.get('image_url')
        
        if image_url:
            logger.info(f'Processing URL: {image_url}')
            file_path = download_image(image_url)
            if not file_path:
                return jsonify({'error': 'Failed to download image'}), 400
        else:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            logger.info(f'Processing uploaded file: {file.filename}')
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            file.save(file_path)

        # Try multiple methods to get location in parallel
        location_data = None
        futures = []
        
        with ThreadPoolExecutor() as executor:
            # Start all detection methods in parallel
            futures.append(executor.submit(get_exif_data, file_path))
            futures.append(executor.submit(detect_location_google_vision, file_path))
            futures.append(executor.submit(detect_location_ai, file_path))
            
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
            return jsonify({
                'error': 'No location data found in image',
                'processing_time': str(datetime.now() - start_time)
            }), 200
        
        # Get detailed location information
        location_name = get_location_name(location_data['latitude'], location_data['longitude'])
        map_path = create_map(location_data['latitude'], location_data['longitude'])
        
        # Get nearby points of interest
        nearby = rg.search((location_data['latitude'], location_data['longitude']))
        
        processing_time = datetime.now() - start_time
        logger.info(f'Request completed in {processing_time}')
        
        return jsonify({
            'latitude': location_data['latitude'],
            'longitude': location_data['longitude'],
            'location': location_name,
            'source': location_data['source'],
            'confidence': location_data.get('confidence', 0.0),
            'nearby': nearby[0] if nearby else None,
            'detected_objects': location_data.get('detected_objects', []),
            'processing_time': str(processing_time),
            'map': bool(map_path)
        })
        
    except Exception as e:
        logger.error(f'Error processing request: {str(e)}')
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary file
        if image_url and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f'Error cleaning up file: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
