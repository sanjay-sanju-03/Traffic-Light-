# Traffic Signal Recognition - Vercel Deployment Guide

A web application for detecting traffic signal colors using OpenCV and machine vision, hosted on Vercel.

## Features

- 🚀 Real-time traffic signal detection (RED, YELLOW, GREEN)
- 🎨 Modern, responsive web interface
- 📸 Image upload and drag-and-drop support
- 🔍 HSV-based color detection with noise reduction
- ⚡ Serverless deployment on Vercel

## Local Development

### Prerequisites
- Python 3.9+
- pip package manager

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run locally:**
```bash
python api/detect.py
```

3. **Open in browser:**
Visit `http://localhost:5000`

## Deployment on Vercel

### Step 1: Prepare Your Repository

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: Traffic signal detection app"
```

### Step 2: Create Vercel Account

1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub, GitLab, or Bitbucket
3. Connect your repository

### Step 3: Deploy

**Option A: Via Vercel Dashboard**
1. Click "New Project"
2. Select your repository
3. Framework: Python
4. Click "Deploy"

**Option B: Via Vercel CLI**
```bash
npm install -g vercel
vercel
```

### Step 4: Configure Environment (if needed)

In Vercel Dashboard → Project Settings → Environment Variables:
- No env variables needed for basic setup

## Project Structure

```
traffic/
├── api/
│   └── detect.py          # Flask app with API endpoints
├── signal_detector.py     # Core detection logic
├── traffic_signal_recognition.py  # Image detection script
├── webcam.py              # Webcam real-time detection
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel configuration
└── README.md             # This file
```

## API Endpoints

### POST `/api/detect`
Detects traffic signal in uploaded image.

**Request (multipart/form-data):**
```
file: <image file>
```

**Response:**
```json
{
  "success": true,
  "signal": "red",
  "signal_text": "RED SIGNAL",
  "image": "data:image/jpeg;base64,...",
  "color_hex": "#ff4444"
}
```

### GET `/api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "Traffic Signal Detector"
}
```

## How It Works

1. **Image Upload**: User uploads or drags an image
2. **Processing**: Image is converted to HSV color space
3. **Detection**: Color masks identify RED, YELLOW, GREEN pixels
4. **Filtering**: Morphological operations remove noise
5. **Classification**: Highest pixel count determines the signal
6. **Response**: Annotated image returned to user

## Color Ranges (HSV)

- **Red**: H: 0-10, 170-180 | S: 120-255 | V: 70-255
- **Yellow**: H: 15-35 | S: 150-255 | V: 150-255
- **Green**: H: 36-85 | S: 100-255 | V: 100-255

## Performance Notes

- Max file size: 16MB
- Processing time: 1-2 seconds per image
- Works best with clear, well-lit traffic lights
- Serverless cold starts: ~5-10 seconds (first request)

## Troubleshooting

**Issue**: "Module not found" errors
- Solution: Ensure `signal_detector.py` is in the root directory

**Issue**: OpenCV errors in production
- Solution: Using `opencv-python-headless` instead of `opencv-python`

**Issue**: Slow first request
- Solution: This is normal for serverless; requests speed up after warm-up

## Future Enhancements

- [ ] Webcam stream support (WebRTC)
- [ ] Batch image processing
- [ ] Model-based detection (CNN/YOLOv8)
- [ ] Analytics dashboard
- [ ] Multi-language support

## License

MIT License - Feel free to use and modify

## Support

For issues or questions:
1. Check the [Vercel documentation](https://vercel.com/docs)
2. Review [Flask documentation](https://flask.palletsprojects.com/)
3. Check OpenCV [color detection guide](https://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html)
