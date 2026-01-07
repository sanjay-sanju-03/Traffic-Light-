# 🚦 Traffic Signal Recognition System

A Python-based traffic signal detection system using OpenCV and HSV color space analysis. Detects RED, YELLOW, and GREEN traffic signals from both webcam streams and uploaded images.

## 📁 Project Structure

```
traffic/
├── src/                                    # Core detection modules (8 files)
│   ├── __init__.py                        # Package initialization
│   ├── signal_detector.py                 # HSV color detection engine (3.6 KB)
│   ├── traffic_signal_recognition.py      # Image file processing (1.6 KB)
│   └── webcam.py                          # Real-time webcam detection (1.9 KB)
│
├── ui/                                    # User interface (2 files)
│   └── dashboard.py                       # Tkinter GUI dashboard (9.8 KB)
│
├── utils/                                 # Utility scripts (4 files)
│   ├── debug_detection.py                 # Debug tool for testing (2.2 KB)
│   ├── generate_images.py                 # Generate test images (1.4 KB)
│   └── __init__.py                        # Package initialization
│
├── images/                                # Sample and test images (7 files)
│   ├── red.jpg                            # Red signal test image (7.3 KB)
│   ├── yellow.jpg                         # Yellow signal test image (5.8 KB)
│   ├── green.jpg                          # Green signal test image (6.7 KB)
│   └── traffic.jpg                        # Real traffic signal image (184 KB)
│
├── main.py                                # Application entry point (275 bytes)
├── README.md                              # Project documentation
└── requirements.txt                       # Project dependencies
```

**Project Stats:**
- 📦 Total Files: 24
- 🐍 Python Files: 8
- 🖼️ Image Files: 4
- 📝 Config Files: 2
- 💾 Total Size: ~0.74 MB

## 🚀 Features

- **Webcam Detection**: Real-time traffic signal detection from webcam
- **Image Upload**: Analyze traffic signals from image files
- **HSV Color Detection**: Accurate color-based signal detection
- **GUI Dashboard**: User-friendly Tkinter interface
- **Debug Mode**: Tools to analyze detection performance
- **Multi-threading**: Non-blocking operations for smooth UI

## 📋 Requirements

```
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.19.0
```

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

## 💻 Installation & Usage

### 1. Install Dependencies
```bash
cd C:\Users\DELL\Desktop\traffic
pip install -r requirements.txt
```

### 2. Run the Dashboard (Main Entry Point)
```bash
python main.py
```

### 3. Choose an Option in Dashboard
- **📷 Webcam Detection**: Real-time signal detection from camera
  - Press 'q' key to exit webcam mode
  
- **📁 Image Upload**: Analyze traffic signals from image files
  - Supports: JPG, PNG, BMP formats
  - Click button to open file dialog and select image

### 4. Debug & Testing
Test detection on sample images with detailed output:
```bash
# From utils directory
python debug_detection.py ../images/red.jpg
python debug_detection.py ../images/yellow.jpg
python debug_detection.py ../images/green.jpg
```

Output includes:
- Pixel counts per color channel
- Color mask visualizations
- Detection accuracy metrics

## 🎨 HSV Color Ranges

The system uses optimized HSV (Hue, Saturation, Value) color ranges for accurate traffic signal detection:

| Signal | Hue Range | Saturation | Value | Notes |
|--------|-----------|-----------|-------|-------|
| RED    | 0-12°, 168-180° | 100-255 | 80-255 | Covers both sides of red in HSV wheel |
| YELLOW | 20-35°    | 100-255 | 80-255 | Pure yellow spectrum |
| GREEN  | 45-90°    | 100-255 | 80-255 | Green spectrum excluding cyan |

**Configuration:** Edit `src/signal_detector.py` to adjust ranges for different lighting conditions.

## 📁 Module Details

### `src/signal_detector.py` (3.6 KB)
**Core detection module using HSV color space analysis.**

**Class: `TrafficSignalDetector`**
- `__init__()`: Initialize detector with color ranges and signal definitions
- `detect(frame)`: Analyzes frame and returns (signal_key, signal_text, color_bgr)
- `get_debug_masks(frame)`: Returns {'red': mask, 'yellow': mask, 'green': mask}

**Key Variables:**
- `RED_LOWER1, RED_UPPER1`: Red color range 1 (0-12°)
- `RED_LOWER2, RED_UPPER2`: Red color range 2 (168-180°)
- `YELLOW_LOWER, YELLOW_UPPER`: Yellow range (20-35°)
- `GREEN_LOWER, GREEN_UPPER`: Green range (45-90°)
- `MIN_PIXELS`: Minimum pixels to detect signal (default: 50)

### `src/webcam.py` (1.9 KB)
**Real-time traffic signal detection from webcam.**

**Function: `main(camera_id=0, display_fps=True, exit_key='q')`**
- Captures video from specified camera
- Displays detected signal in real-time overlay
- Shows frame count when display_fps=True
- Exit on pressing specified key (default: 'q')

**Returns:** True on successful completion, False on error

### `src/traffic_signal_recognition.py` (1.6 KB)
**Static image file processing and analysis.**

**Function: `main(image_path="traffic.jpg", debug=False)`**
- Reads and processes single image file
- Returns detected signal text and applies overlay
- Optional debug mode displays individual color masks
- Accepts command-line arguments: image_path and --debug flag

**Returns:** True on success, False on file not found

### `ui/dashboard.py` (9.8 KB)
**Tkinter-based graphical user interface dashboard.**

**Class: `TrafficSignalDashboard`**
- Modern GUI with professional styling
- Two main operation modes:
  1. **Webcam Detection**: Launch real-time detection
  2. **Image Upload**: File dialog for image selection
- Status bar showing current operation
- Information panel with feature list
- Multi-threaded operations (non-blocking)
- Error handling with user-friendly messages

**Features:**
- Color-coded interface (dark header, light content)
- Real-time status updates
- Thread-based processing to prevent UI freezing

### `utils/debug_detection.py` (2.2 KB)
**Debug and analysis tool for detection accuracy.**

**Function: `debug_image(image_path)`**
- Displays three color mask windows (Red, Yellow, Green)
- Prints pixel count statistics for each color
- Shows detection decision logic
- Useful for tuning HSV ranges

**Usage:** `python debug_detection.py ../images/signal.jpg`

**Output:**
```
📊 Detection Results for: images/red.jpg
==================================================
🔴 Red pixels:    22810
🟡 Yellow pixels: 0
🟢 Green pixels:  0
==================================================
✅ Detected: RED SIGNAL
==================================================
```

### `utils/generate_images.py` (1.4 KB)
**Generate synthetic test images for development.**

Creates sample traffic signal images:
- `red.jpg`: Red circle on gray background
- `yellow.jpg`: Yellow circle on gray background
- `green.jpg`: Green circle on gray background

**Usage:** `python generate_images.py`

**Location:** Images are saved to `images/` directory

## 🔧 Configuration & Customization

### Adjust Color Detection Ranges
Edit `src/signal_detector.py` to modify HSV ranges:

```python
# Example: Make red detection more sensitive
RED_LOWER1 = np.array([0, 80, 100])      # Lower saturation threshold
RED_UPPER1 = np.array([12, 255, 255])
RED_LOWER2 = np.array([168, 80, 100])
RED_UPPER2 = np.array([180, 255, 255])
```

### Change Detection Sensitivity
Adjust minimum pixel threshold:

```python
MIN_PIXELS = 50   # Lower = more sensitive, Higher = less sensitive
```

### Webcam Configuration
Edit camera selection in `main()` function:

```python
# Use different camera
webcam_detection(camera_id=1)  # For secondary camera
```

### Image Processing Size
Modify resize dimensions:

```python
image = cv2.resize(image, (500, 700))  # Change dimensions as needed
```

## 🐛 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Webcam not accessible** | Camera in use or permissions denied | Check system camera permissions, close other apps |
| **Wrong color detection** | HSV ranges not optimal for lighting | Use debug_detection.py to check HSV values, adjust ranges |
| **Image not reading** | Unsupported format or corrupted file | Use JPG, PNG, or BMP; verify file integrity |
| **GUI freezing** | Long operation on main thread | Wait for operation to complete, improvements in progress |
| **Detection too sensitive** | MIN_PIXELS threshold too low | Increase MIN_PIXELS value in signal_detector.py |
| **Detection too insensitive** | MIN_PIXELS threshold too high | Decrease MIN_PIXELS value or improve lighting |

### Quick Fix Checklist
- [ ] Verify camera/webcam works in other applications
- [ ] Test with sample images first (red.jpg, yellow.jpg, green.jpg)
- [ ] Check image file format and size
- [ ] Run debug_detection.py to analyze HSV values
- [ ] Ensure proper lighting conditions for webcam

## 📝 Example Output

### Webcam Detection Console Output
```
Press 'q' to exit
Processed 10 frames | Last signal: RED SIGNAL
Processed 20 frames | Last signal: YELLOW SIGNAL
Processed 30 frames | Last signal: GREEN SIGNAL
Exiting... (processed 45 frames)
```

### Debug Detection Tool Output
```
📊 Detection Results for: images/red.jpg
==================================================
🔴 Red pixels:    22810
🟡 Yellow pixels: 0
🟢 Green pixels:  0
==================================================
✅ Detected: RED SIGNAL
==================================================
```

### Dashboard GUI
- Professional dark header with "🚦 Traffic Signal Recognition"
- Two clickable option cards: Webcam and Image Upload
- Real-time status bar at bottom
- Information panel with feature highlights

## 📚 Additional Resources

### Color Space Information
- **HSV vs RGB**: HSV is more intuitive for color detection in varying lighting
- **Hue**: 0-180° (OpenCV uses 0-180 scale, not 0-360)
- **Saturation**: 0-255 (color intensity)
- **Value**: 0-255 (brightness)

### OpenCV Functions Used
- `cv2.cvtColor()`: Color space conversion (BGR to HSV)
- `cv2.inRange()`: Extract color masks based on range
- `cv2.morphologyEx()`: Noise reduction and cleanup
- `cv2.countNonZero()`: Count pixels matching criteria
- `cv2.putText()`: Add text overlays

### Performance Tips
- Use webcam resolution 640x480 for balanced speed/accuracy
- Ensure adequate lighting for better detection
- Keep environment consistent for stable results
- Use generate_images.py to create test data

## 👥 Project Information

**Traffic Signal Recognition System v1.0.0**

- **Language**: Python 3.7+
- **Framework**: OpenCV (Computer Vision)
- **UI**: Tkinter (GUI)
- **Platform**: Windows, Linux, macOS
- **License**: Open Source

## 🚀 Future Enhancements

- [ ] Deep learning-based detection (CNN models)
- [ ] Real-time statistics dashboard
- [ ] Video file processing support
- [ ] Export detection results to CSV
- [ ] Mobile app version
- [ ] Multi-signal detection (multiple lights simultaneously)
- [ ] Predictive state changes
- [ ] Performance metrics logging

---

**Questions or Issues?** 
1. Check troubleshooting section above
2. Run debug_detection.py for detailed analysis
3. Review HSV color ranges in signal_detector.py
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

**Last Updated:** January 5, 2026 | **Status:** ✅ Production Ready
#   T r a f f i c - L i g h t -  
 