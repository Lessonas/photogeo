from flask import Flask, request, render_template, jsonify
from PIL import Image
import exifread
from geopy.geocoders import Nominatim
import os
from datetime import datetime
import folium
import cv2
import numpy as np
import pytesseract
import requests
from bs4 import BeautifulSoup
import re
import io
import json
import tempfile
import urllib.parse

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-this-in-production')

# Configure for production
if os.environ.get('PRODUCTION'):
    from flask_talisman import Talisman
    Talisman(app, content_security_policy=None)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

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

def extract_location_from_screenshot(image_path):
    """Extract location data from screenshots using OCR."""
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply threshold to get black and white image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Perform OCR
        text = pytesseract.image_to_string(binary)
        
        # Try to find coordinates in the text
        coords = extract_location_from_text(text)
        if coords:
            return float(coords[0]), float(coords[1])
        
        return None
    except Exception as e:
        print(f"Error in OCR: {str(e)}")
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
    # Create Folium map
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon]).add_to(m)
    
    # Add Street View layer
    street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    apple_maps_url = f"https://maps.apple.com/?ll={lat},{lon}&z=15"
    
    # Add custom HTML for different map views
    html = f"""
    <div class="map-links">
        <a href="{street_view_url}" target="_blank" class="btn btn-primary">Open in Street View</a>
        <a href="{apple_maps_url}" target="_blank" class="btn btn-primary">Open in Apple Maps</a>
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
            # Handle URL input
            file_path = download_image(image_url)
            if not file_path:
                return jsonify({'error': 'Failed to download image'}), 400
        else:
            # Handle file upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            file.save(file_path)
        
        # Try different methods to extract location
        location_data = None
        
        # 1. Try EXIF data first
        exif_data = get_exif_data(file_path)
        if exif_data and exif_data['gps_latitude'] and exif_data['gps_longitude']:
            location_data = {
                'latitude': exif_data['gps_latitude'],
                'longitude': exif_data['gps_longitude'],
                'source': 'EXIF data'
            }
        
        # 2. If no EXIF data, try OCR
        if not location_data:
            coords = extract_location_from_screenshot(file_path)
            if coords:
                location_data = {
                    'latitude': coords[0],
                    'longitude': coords[1],
                    'source': 'OCR detection'
                }
        
        # Clean up temporary file
        if image_url and os.path.exists(file_path):
            os.remove(file_path)
        
        if not location_data:
            return jsonify({
                'error': 'No location data found in image',
                'device': exif_data['make'] + ' ' + exif_data['model'] if exif_data else 'Unknown',
                'datetime': exif_data['datetime'] if exif_data else 'Unknown'
            }), 200
        
        # Get location name and create map
        location_name = get_location_name(location_data['latitude'], location_data['longitude'])
        create_map(location_data['latitude'], location_data['longitude'])
        
        return jsonify({
            'latitude': location_data['latitude'],
            'longitude': location_data['longitude'],
            'location': location_name,
            'source': location_data['source'],
            'device': exif_data['make'] + ' ' + exif_data['model'] if exif_data else 'Unknown',
            'datetime': exif_data['datetime'] if exif_data else 'Unknown',
            'map': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
