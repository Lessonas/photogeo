# Photo Geolocator

A web application that extracts and displays geolocation data from photos. This application can handle photos from any device and provides detailed location information including interactive maps.

## Features

- Drag and drop interface for easy photo upload
- Extracts EXIF data including GPS coordinates
- Shows device information (make/model)
- Displays photo capture date and time
- Reverse geocoding to show human-readable location names
- Interactive map display of photo location
- Supports photos from any device that stores GPS data in EXIF format
- OCR capability for extracting location from screenshots
- Support for processing images from URLs
- Direct links to Google Street View and Apple Maps

## Local Development

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to `http://localhost:5000`

## Deployment

### Deploy to Render (Free)

1. Fork this repository to your GitHub account

2. Sign up for a free account at [Render.com](https://render.com)

3. Create a new Web Service:
   - Connect your GitHub repository
   - Select the Python environment
   - The build and start commands are already configured in `render.yaml`
   - Deploy!

### Deploy to Heroku (Free with Student Pack)

1. Install the Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Set buildpacks:
```bash
heroku buildpacks:add --index 1 heroku/python
heroku buildpacks:add --index 2 https://github.com/heroku/heroku-buildpack-apt
```

5. Deploy:
```bash
git push heroku main
```

### Deploy to Google Cloud Run (Free Tier)

1. Install the Google Cloud SDK
2. Initialize and configure your project:
```bash
gcloud init
```

3. Build and deploy:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/photo-geolocator
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/photo-geolocator --platform managed
```

## Security Notes

- The application uses secure headers with Flask-Talisman in production
- File uploads are restricted to images and have a size limit
- Temporary files are automatically cleaned up
- HTTPS is enforced in production
- API keys and secrets are managed through environment variables

## Technical Details

- Built with Flask
- Uses Pillow and exifread for image processing
- Geopy for reverse geocoding
- Folium for map generation
- Bootstrap for responsive UI
- Tesseract OCR for screenshot text extraction
- Docker support for containerized deployment

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
