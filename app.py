from flask import Flask, request, render_template, jsonify
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-this-in-production')

# Initialize AI models
try:
    # Load landmark detection model
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # Initialize Google Cloud Vision client
    vision_client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Warning: AI models not loaded: {str(e)}")

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
        print(f"Google Vision API error: {str(e)}")
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
        print(f"AI model error: {str(e)}")
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
        print(f"Context analysis error: {str(e)}")
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
        print(f"Error downloading image: {str(e)}")
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
        print(f"Error reading EXIF: {str(e)}")
        return None

def convert_to_degrees(values):
    """Convert GPS coordinates to degrees."""
    d = float(values[0].num) / float(values[0].den)
    m = float(values[1].num) / float(values[1].den)
    s = float(values[2].num) / float(values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

def get_location_name(lat, lon):
    """Get location name from coordinates using reverse geocoding."""
    try:
        geolocator = Nominatim(user_agent="photo_geolocator")
        location = geolocator.reverse(f"{lat}, {lon}", language='en')
        return location.address if location else "Location not found"
    except:
        return "Location lookup failed"

def create_map(lat, lon):
    """Create maps with different views."""
    # Create Folium map with minimal features for faster loading
    m = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        prefer_canvas=True,
        disable_3d=True,
        tiles='CartoDB positron'  # Using a lighter tile set
    )
    
    # Add a simple marker
    folium.Marker(
        [lat, lon],
        popup='Location',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add custom HTML for different map views
    street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    apple_maps_url = f"https://maps.apple.com/?ll={lat},{lon}&z=15"
    
    html = f"""
    <div class="map-links" style="text-align: center; padding: 10px;">
        <a href="{street_view_url}" target="_blank" class="btn btn-primary" style="margin: 5px;">Street View</a>
        <a href="{apple_maps_url}" target="_blank" class="btn btn-primary" style="margin: 5px;">Apple Maps</a>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(html))
    map_path = os.path.join('static', 'map.html')
    m.save(map_path)
    return map_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = None
        image_url = request.form.get('image_url')
        
        if image_url:
            file_path = download_image(image_url)
            if not file_path:
                return jsonify({'error': 'Failed to download image'}), 400
        else:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            file.save(file_path)

        # Try multiple methods to get location
        location_data = None
        
        # 1. Try EXIF data first
        exif_data = get_exif_data(file_path)
        if exif_data and exif_data['gps_latitude'] and exif_data['gps_longitude']:
            location_data = {
                'latitude': exif_data['gps_latitude'],
                'longitude': exif_data['gps_longitude'],
                'source': 'EXIF data',
                'confidence': 1.0
            }
        
        # 2. Try Google Cloud Vision API
        if not location_data:
            vision_data = detect_location_google_vision(file_path)
            if vision_data and 'latitude' in vision_data:
                location_data = vision_data
        
        # 3. Try AI-based detection
        if not location_data:
            ai_predictions = detect_location_ai(file_path)
            if ai_predictions:
                context_location = analyze_image_context(ai_predictions)
                if context_location:
                    location_data = context_location
        
        if not location_data:
            return jsonify({
                'error': 'No location data found in image',
                'device': exif_data['make'] + ' ' + exif_data['model'] if exif_data else 'Unknown',
                'datetime': exif_data['datetime'] if exif_data else 'Unknown'
            }), 200
        
        # Get detailed location information
        location_name = get_location_name(location_data['latitude'], location_data['longitude'])
        create_map(location_data['latitude'], location_data['longitude'])
        
        # Get nearby points of interest
        nearby = rg.search((location_data['latitude'], location_data['longitude']))
        
        return jsonify({
            'latitude': location_data['latitude'],
            'longitude': location_data['longitude'],
            'location': location_name,
            'source': location_data['source'],
            'confidence': location_data.get('confidence', 0.0),
            'device': exif_data['make'] + ' ' + exif_data['model'] if exif_data else 'Unknown',
            'datetime': exif_data['datetime'] if exif_data else 'Unknown',
            'nearby': nearby[0] if nearby else None,
            'detected_objects': location_data.get('detected_objects', []),
            'map': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary file
        if image_url and file_path and os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
