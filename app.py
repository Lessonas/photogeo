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
        
        # Try EXIF data
        exif_data = get_exif_data(file_path)
        if not exif_data or not (exif_data['gps_latitude'] and exif_data['gps_longitude']):
            return jsonify({
                'error': 'No location data found in image',
                'device': exif_data['make'] + ' ' + exif_data['model'] if exif_data else 'Unknown',
                'datetime': exif_data['datetime'] if exif_data else 'Unknown'
            }), 200
        
        # Get location name and create map
        location_name = get_location_name(exif_data['gps_latitude'], exif_data['gps_longitude'])
        create_map(exif_data['gps_latitude'], exif_data['gps_longitude'])
        
        return jsonify({
            'latitude': exif_data['gps_latitude'],
            'longitude': exif_data['gps_longitude'],
            'location': location_name,
            'source': 'EXIF data',
            'device': exif_data['make'] + ' ' + exif_data['model'] if exif_data else 'Unknown',
            'datetime': exif_data['datetime'] if exif_data else 'Unknown',
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
