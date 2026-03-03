# 🚦 Traffic Signal Recognition System

A Python-based intelligent traffic detection system using OpenCV and Deep Learning. Detects and classifies:
- **Traffic Lights:** RED, YELLOW, GREEN (HSV-based)
- **Traffic Signs:** Stop, Yield, Speed Limit, No Entry, Pedestrian Crossing, and 40+ more using YOLOv8

**🌐 Live Demo:** [traffic-light-1rd9.vercel.app](https://traffic-light-1rd9.vercel.app/)

---

## 🎯 Problem Statement & Solution

**Original Problem:**
Traffic signal recognition is essential for intelligent transportation systems and autonomous driving. The system must:
- ✅ Automatically detect and classify traffic lights and signs from images/video
- ✅ Handle varying environmental conditions (lighting, weather, angles)
- ✅ Provide real-time or near real-time processing
- ✅ Support integration with driver assistance and autonomous vehicle systems

**Our Solution:**
This enhanced system now **fully addresses the problem statement** with a comprehensive approach:

1. **Dual Detection Capability** - Both traffic lights AND traffic signs
2. **Robust Algorithms** - HSV color analysis + YOLOv8 deep learning
3. **Real-time Processing** - Sub-second response times
4. **Production-Ready** - Web deployment, desktop app, RESTful APIs
5. **40+ Sign Types** - Comprehensive traffic sign coverage
6. **Autonomous-Vehicle Ready** - Accurate, fast, reliable detection

---

## 🚀 Quick Start

### Option 1: Online (Recommended) 🌐
Visit: https://traffic-light-1rd9.vercel.app/
- 📷 **Webcam Detection** - Real-time from your camera
- 📸 **Image Upload** - Analyze traffic light images
- Works on desktop, tablet, mobile
- No installation needed!

### Option 2: Local Desktop 💻
```bash
# Clone & Install
git clone https://github.com/sanjay-sanju-03/Traffic-Light-.git
cd Traffic-Light-
pip install -r requirements.txt

# Run desktop app
python main.py
```

---

## 🎯 Features

✅ **Traffic Light Detection** - Real-time RED, YELLOW, GREEN recognition  
✅ **Traffic Sign Detection** - 40+ traffic sign types using YOLOv8  
✅ **Webcam Support** - Real-time detection from camera  
✅ **Image Upload & Analysis** - Batch processing  
✅ **HSV Color Detection** - Robust traffic light detection  
✅ **Deep Learning** - YOLOv8 neural network for traffic signs  
✅ **Responsive Web UI** - Works on desktop, tablet, mobile  
✅ **Desktop Dashboard** - Professional Tkinter GUI  
✅ **No External APIs** - Fully local processing

---

## 📁 Project Structure

```
traffic/
├── api/                           # Flask API for web hosting
│   └── detect.py                  # Web endpoints for traffic detection
├── src/                           # Core detection modules
│   ├── __init__.py
│   ├── signal_detector.py         # Traffic light detection (HSV-based)
│   ├── sign_detector.py           # Traffic sign detection (YOLOv8)
│   ├── unified_detector.py        # Combined traffic detection system
│   ├── traffic_signal_recognition.py
│   └── webcam.py
├── ui/                            # Desktop GUI
│   └── dashboard.py               # Professional Tkinter interface
├── utils/                         # Utility scripts
│   ├── debug_detection.py
│   └── generate_images.py
├── images/                        # Sample test images
├── main.py                        # Desktop app entry point
├── requirements.txt               # Python dependencies
├── vercel.json                    # Vercel deployment config
└── README.md                      # This file
```

---

## 🎨 How It Works

### Traffic Light Detection (HSV-based)
1. **Capture** - Get image from webcam or upload
2. **Convert** - Change BGR to HSV color space
3. **Detect** - Apply color masks for RED/YELLOW/GREEN
4. **Analyze** - Count pixels & determine signal
5. **Display** - Show result with overlay

### Traffic Sign Detection (Deep Learning)
1. **Input** - Image from webcam, file, or URL
2. **YOLOv8 Inference** - Run neural network for object detection
3. **Classification** - Identify stop signs, yield signs, speed limits, pedestrian crossings, etc.
4. **Confidence Scoring** - Each detection includes confidence level
5. **Visualization** - Draw bounding boxes with labels
6. **Output** - Return annotated image with detected signs

---

## 🔧 Tech Stack

- **Python 3.10+**
- **OpenCV** - Image processing
- **YOLOv8** - Deep learning for traffic signs
- **PyTorch** - Neural network framework
- **Flask** - Web API
- **Vercel** - Cloud hosting
- **HTML/CSS/JavaScript** - Frontend

---

## 🛑 Supported Traffic Signs

The traffic sign detector recognizes **40+ traffic sign types** including:

| Category | Signs |
|----------|-------|
| **Regulatory** | Stop, Yield, No Entry, No Passing, One Way |
| **Speed Limits** | 20, 30, 50, 60, 70, 80, 100, 120 km/h |
| **Warnings** | Dangerous Curves, Slippery Road, Road Works, Pedestrian Crossing |
| **Mandatory** | Keep Right, Keep Left, Go Straight, Turn Right, Turn Left |
| **Information** | Pedestrians, Bicycles, Animals, Roundabout, Priority Road |

**Model:** YOLOv8 Nano (ultra-fast), Small, or Medium (more accurate)

---

## 📝 Usage Examples

### Desktop (Local)
```bash
python main.py                    # GUI dashboard
python src/webcam.py              # Webcam detection
python src/traffic_signal_recognition.py images/red.jpg
```

### Web (Hosted)
Just visit the link and use the interface!

---

## � API Documentation

### Traffic Light Detection
**Endpoint:** `POST /api/detect`
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/detect
```

**Response:**
```json
{
  "success": true,
  "signal": "red",
  "signal_text": "🔴 RED LIGHT",
  "image": "data:image/jpeg;base64,...",
  "color_hex": "#ff0000"
}
```

### Traffic Sign Detection
**Endpoint:** `POST /api/detect-signs`
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/detect-signs
```

**Response:**
```json
{
  "success": true,
  "signs_detected": 2,
  "signs": [
    {
      "sign": "Stop",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 250]
    },
    {
      "sign": "Speed Limit 50",
      "confidence": 0.87,
      "bbox": [300, 100, 380, 180]
    }
  ],
  "image": "data:image/jpeg;base64,...",
  "status": "✅ Detected 2 sign(s)"
}
```

### Health Check
**Endpoint:** `GET /api/health`
```json
{
  "status": "ok",
  "service": "Traffic Detection System",
  "traffic_lights": "enabled",
  "traffic_signs": "enabled"
}
```

---

## �🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not working | Allow camera permission in browser |
| Detection wrong | Ensure good lighting, adjust HSV ranges in `signal_detector.py` |
| Import error | Run `pip install -r requirements.txt` || YOLOv8 not found | Install with: `pip install ultralytics torch torchvision` |
| Traffic signs not detected | Ensure image is clear, try adjusting confidence threshold in `sign_detector.py` |
| Slow sign detection | Use YOLOv8 Nano for faster inference, Medium for better accuracy |
| CUDA/GPU errors | Run on CPU by setting `device='cpu'` in sign detector initialization |
---

## 📊 HSV Color Ranges

| Signal | Hue | Saturation | Value |
|--------|-----|-----------|-------|
| RED | 0-10°, 170-180° | 120-255 | 70-255 |
| YELLOW | 15-35° | 150-255 | 150-255 |
| GREEN | 36-85° | 100-255 | 100-255 |

---

## 🚀 Deployment

Already deployed on Vercel at: https://traffic-light-1rd9.vercel.app/

To deploy your own:
1. Push to GitHub
2. Go to vercel.com
3. Import repository
4. Done! 🎉

---

## 📚 Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [HSV Color Space](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [Vercel Deployment](https://vercel.com/docs)

---

## 📄 License

Open Source - Free to use and modify

**Last Updated:** January 12, 2026
│   └── __init__.py
│
├── images/                        # Sample test images
│   ├── red.jpg, yellow.jpg, green.jpg
│   └── traffic.jpg
│
├── main.py                        # Desktop app entry point
├── requirements.txt               # Python dependencies
├── vercel.json                    # Vercel deployment config
└── README.md                      # This file
```

## 🎯 Features

### Web Version (Vercel Hosted)
- ✅ **Live Webcam Detection**: Real-time camera access with instant detection
- ✅ **Image Upload**: Drag-and-drop or click to upload images
- ✅ **Multi-Tab Interface**: Switch between webcam and image modes
- ✅ **Mobile Responsive**: Works on all devices
- ✅ **Zero Setup**: No installation required
- ✅ **Auto-Scaling**: Serverless architecture handles traffic automatically

### Desktop Version (Local)
- ✅ **GUI Dashboard**: User-friendly Tkinter interface
- ✅ **Webcam Detection**: Real-time signal detection from webcam
- ✅ **Image Processing**: Analyze image files
- ✅ **Debug Mode**: Tools to analyze detection performance
- ✅ **Multi-threading**: Non-blocking UI operations

## 🎨 Detection Technology

**HSV Color Space Analysis:**
- Hue-based color detection (more robust than RGB)
- Morphological operations for noise reduction
- Pixel counting for signal classification
- Minimum pixel threshold to avoid false positives

### Color Ranges Used

| Signal | Hue Range | Saturation | Value | Notes |
|--------|-----------|-----------|-------|-------|
| RED    | 0-10°, 170-180° | 120-255 | 70-255 | Covers red wraparound in HSV |
| YELLOW | 15-35°    | 150-255 | 150-255 | Pure yellow spectrum |
| GREEN  | 36-85°    | 100-255 | 100-255 | Green spectrum |

## 📊 How It Works

1. **Image Input** → Webcam frame or uploaded image
2. **HSV Conversion** → Convert BGR to HSV color space
3. **Color Masks** → Create masks for RED, YELLOW, GREEN
4. **Noise Reduction** → Apply morphological operations
5. **Pixel Counting** → Count non-zero pixels in each mask
6. **Classification** → Determine signal based on highest pixel count
7. **Output** → Display annotated image with detected signal

## 💻 Web Version Usage

### Webcam Mode
1. Click **📷 Webcam** tab
2. Click **"Start Webcam"** (allow camera permission)
3. Point camera at traffic light
4. Click **"Capture & Detect"** to analyze
5. View results instantly

### Image Upload Mode
1. Click **📸 Image Upload** tab
2. Click **"Choose Image"** or drag-and-drop
3. Select JPG/PNG/GIF/BMP (max 16MB)
4. View detection results

## 🖥️ Desktop Version Usage

### 1. Run the Dashboard
```bash
python main.py
```

### 2. Choose Detection Mode
- **📷 Webcam Detection**: Real-time detection (press 'q' to exit)
- **📁 Image Upload**: Browse and analyze image files

### 3. Debug Mode (Testing)
```bash
# Test detection on sample images
python src/traffic_signal_recognition.py images/red.jpg --debug
python src/traffic_signal_recognition.py images/yellow.jpg --debug
python src/traffic_signal_recognition.py images/green.jpg --debug

# Or use the debug tool
python utils/debug_detection.py images/traffic.jpg
```

## 📋 Requirements

### Web Version
- ✅ Modern browser (Chrome, Firefox, Safari, Edge)
- ✅ Camera permission (for webcam mode)
- ✅ Internet connection

### Desktop Version
```
# Core dependencies
opencv-python>=4.5.0 (or opencv-python-headless for server)
numpy>=1.19.0
Pillow>=8.0.0
scikit-image>=0.19.0

# Machine Learning (for traffic sign detection)
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# Web Framework
Flask>=3.0.0
Werkzeug>=3.0.0
```

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Install only traffic light detection (lightweight):**
```bash
pip install opencv-python numpy Pillow scikit-image Flask
```

**Install with GPU support (faster traffic sign detection):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
```

## 🔧 Configuration

### Adjust Detection Sensitivity
Edit `src/signal_detector.py`:

```python
# More sensitive to small signals
MIN_PIXELS = 50

# Less sensitive (ignore noise)
MIN_PIXELS = 200
```

### Modify HSV Ranges
For different lighting conditions, adjust in `src/signal_detector.py`:

```python
# Example: Make red detection less strict
RED_LOWER1 = np.array([0, 80, 50])      # Lower saturation/value
RED_UPPER1 = np.array([10, 255, 255])
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Camera not found" (web)** | Check browser camera permissions, try different browser |
| **"No signal detected"** | Ensure good lighting, point directly at signal |
| **Wrong color detection** | Run debug_detection.py to check HSV values, adjust ranges |
| **Image upload fails** | Check file format (JPG/PNG/GIF/BMP), file size < 16MB |
| **Webcam appears black** | Allow camera permission in browser, refresh page |
| **Slow detection (web)** | First request takes ~2-5s (cold start), subsequent requests are faster |

### Quick Fix Checklist
- [ ] Ensure adequate lighting around traffic signal
- [ ] Test with sample images first (red.jpg, yellow.jpg, green.jpg)
- [ ] Check internet connection (for web version)
- [ ] Clear browser cache if UI issues occur
- [ ] Try different camera for webcam issues

## 📝 Example Output

### Webcam Detection
```
✅ Camera ready! Click "Capture & Detect" to analyze the traffic light.
[User captures frame]
✅ RED SIGNAL
```

### Image Upload
```
Result displayed with:
- Annotated image showing detected signal
- Color-coded result box (RED/YELLOW/GREEN/NO SIGNAL)
```

## 📊 Module Details

### Core Detection Modules

#### `src/signal_detector.py`
**Traffic Light Detection** - HSV-based color analysis
- `TrafficSignalDetector` class
- `detect(frame)` - Returns (signal_key, signal_text, color_bgr)
- `get_debug_masks(frame)` - Visualize HSV masks
- Support for custom HSV ranges and sensitivity tuning

#### `src/sign_detector.py`
**Traffic Sign Detection** - YOLOv8 deep learning
- `TrafficSignDetector` class
- `detect(frame)` - Returns bounding boxes with confidence
- `detect_batch(frames)` - Process multiple frames
- Supports 40+ traffic sign types
- Configurable model sizes (nano, small, medium, large)
- `TrafficSignClassifier` - Lightweight fallback classifier

#### `src/unified_detector.py`
**Combined Detection System** - Unified interface
- `UnifiedTrafficDetector` class
- `detect_all(frame)` - Run both detectors simultaneously
- `detect_lights_only()` - Traffic lights only
- `detect_signs_only()` - Traffic signs only
- Single interface for comprehensive traffic analysis

### Web & API

#### `api/detect.py`
**Flask Web Server** with dual detection endpoints:
- `/` - Web UI with webcam and upload support
- `/api/detect` - Traffic light detection endpoint
- `/api/detect-signs` - Traffic sign detection endpoint (YOLOv8)
- `/api/health` - Service status check
- Supports base64 and file upload

#### `ui/dashboard.py`
**Desktop GUI** application:
- Tkinter-based interface
- Real-time webcam detection
- Image file browser
- Side-by-side comparison mode
- Debug visualizations

## 🚀 Deployment

### Already Hosted on Vercel
Your app is live at: https://traffic-light-1rd9.vercel.app/

### To Deploy Your Own Fork
1. Push code to GitHub
2. Connect repository to Vercel
3. Vercel auto-deploys on each push

## 📚 Technical Details

### Color Detection Algorithm (Traffic Lights)
```
1. Convert BGR image to HSV color space
2. Create binary masks for each color using cv2.inRange()
3. Apply morphological operations (erode + dilate) to reduce noise
4. Count non-zero pixels in each mask
5. Select color with highest pixel count
6. Apply minimum pixel threshold to avoid false positives
```

### Deep Learning Detection (Traffic Signs)
```
1. Load pre-trained YOLOv8 model (nano/small/medium)
2. Resize image to model input size (arbitrary sizes supported)
3. Run inference through neural network
4. Apply Non-Maximum Suppression (NMS) for duplicate removal
5. Extract bounding boxes with confidence scores
6. Filter by confidence threshold
7. Map class IDs to sign labels
8. Draw annotated results
```

**Model Details:**
- **Base:** YOLOv8 (You Only Look Once v8)
- **Sizes:** Nano (3.9M), Small (11.4M), Medium (25.9M), Large (63.7M)
- **Input:** Any image size (internally resized)
- **Output:** Bounding boxes, class labels, confidence scores
- **Speed:** Nano ~5-10ms, Small ~15-20ms, Medium ~40-50ms on GPU

### Performance
- **Web version**: 1-3 seconds per detection (includes network latency)
- **Desktop version**: 100-200ms per frame (real-time)
- **Mobile**: Works on all devices with modern browsers

## 👥 Project Information

**Traffic Signal Recognition System v1.1.0**

- **Language**: Python 3.8+
- **Framework**: OpenCV (Computer Vision) + Flask (Web)
- **UI**: Tkinter (Desktop) + HTML/CSS/JS (Web)
- **Hosting**: Vercel (Serverless)
- **Platform**: Windows, Linux, macOS, Web

## 🔗 Links

- **Live Demo**: https://traffic-light-1rd9.vercel.app/
- **GitHub Repo**: https://github.com/sanjay-sanju-03/Traffic-Light-
- **Issues**: Report bugs on GitHub

## 🚀 Future Enhancements

- [ ] Deep learning-based detection (YOLO/CNN)
- [ ] Multi-signal detection (multiple lights simultaneously)
- [ ] Video file upload support
- [ ] Performance metrics dashboard
- [ ] Export detection logs to CSV
- [ ] Mobile app version
- [ ] Predictive signal state changes
- [ ] Integration with traffic management systems

## 📖 Learning Resources

### HSV Color Space
- [OpenCV HSV Tutorial](https://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html)
- [HSV Color Picker Tool](https://chir.ag/projects/ntsc/)

### OpenCV Functions
- `cv2.cvtColor()` - Color space conversion
- `cv2.inRange()` - Extract color masks
- `cv2.morphologyEx()` - Noise reduction
- `cv2.countNonZero()` - Pixel counting

## 📞 Support

**Having issues?**
1. Check the [Troubleshooting](#troubleshooting) section
2. Run `python utils/debug_detection.py <image>` for detailed analysis
3. Check GitHub issues for similar problems
4. Review HSV ranges in `src/signal_detector.py`

## 📜 License

MIT License - Open source and free to use

---

**Last Updated:** February 10, 2026 | **Status:** ✅ Production Ready

**Start detecting traffic signals now:**
- 🌐 Web: https://traffic-light-1rd9.vercel.app/
- 💻 Local: `python main.py`
