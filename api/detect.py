from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

# Import detector from src directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from signal_detector import TrafficSignalDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

detector = TrafficSignalDetector()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(cv_image):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', cv_image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/api/detect', methods=['POST'])
def detect_signal():
    """
    API endpoint for traffic signal detection.
    Accepts image file or base64 encoded image.
    Returns signal type and annotated image.
    """
    try:
        # Check if file is in request
        if 'file' not in request.files and 'image' not in request.form:
            return jsonify({'error': 'No image provided'}), 400
        
        image = None
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
            
            # Read image file
            file_stream = file.read()
            nparr = np.frombuffer(file_stream, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Handle base64 image
        elif 'image' in request.form:
            try:
                base64_str = request.form['image']
                if base64_str.startswith('data:image'):
                    base64_str = base64_str.split(',')[1]
                
                image_data = base64.b64decode(base64_str)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({'error': f'Invalid base64 image: {str(e)}'}), 400
        
        if image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Resize image (maintain aspect ratio if needed)
        height, width = image.shape[:2]
        if width > 800 or height > 600:
            scale = min(800/width, 600/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Detect signal
        signal_key, signal_text, color = detector.detect(image)
        
        # Annotate image with result
        cv2.putText(image, signal_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Convert to base64
        result_image_b64 = image_to_base64(image)
        
        # Return response
        return jsonify({
            'success': True,
            'signal': signal_key,
            'signal_text': signal_text,
            'image': result_image_b64,
            'color_hex': '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])  # Convert BGR to RGB hex
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'Traffic Signal Detector'}), 200

@app.route('/', methods=['GET'])
def index():
    """Serve the web interface with webcam support."""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Traffic Signal Recognition</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
                font-size: 28px;
            }
            
            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 30px;
                font-size: 14px;
            }
            
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
                border-bottom: 2px solid #e0e0e0;
            }
            
            .tab-button {
                flex: 1;
                padding: 15px;
                border: none;
                background: none;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                color: #999;
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
            }
            
            .tab-button.active {
                color: #667eea;
                border-bottom-color: #667eea;
            }
            
            .tab-button:hover {
                color: #764ba2;
            }
            
            .tab-content {
                display: none;
            }
            
            .tab-content.active {
                display: block;
            }
            
            /* Image Upload Styles */
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #f8f9ff;
            }
            
            .upload-area:hover {
                border-color: #764ba2;
                background: #f0f2ff;
            }
            
            .upload-area.dragover {
                border-color: #764ba2;
                background: #e8ebff;
                transform: scale(1.02);
            }
            
            .upload-icon {
                font-size: 48px;
                margin-bottom: 15px;
            }
            
            .upload-text {
                color: #333;
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 5px;
            }
            
            .upload-subtext {
                color: #999;
                font-size: 13px;
            }
            
            input[type="file"] {
                display: none;
            }
            
            /* Webcam Styles */
            .webcam-container {
                text-align: center;
            }
            
            video {
                width: 100%;
                max-width: 600px;
                border-radius: 8px;
                background: #000;
                margin-bottom: 20px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            
            canvas {
                display: none;
            }
            
            .button-group {
                display: flex;
                gap: 10px;
                margin-top: 20px;
                justify-content: center;
                flex-wrap: wrap;
            }
            
            button {
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            
            .btn-primary:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            
            .btn-secondary {
                background: #f0f0f0;
                color: #333;
            }
            
            .btn-secondary:hover {
                background: #e0e0e0;
            }
            
            .btn-danger {
                background: #ff4444;
                color: white;
            }
            
            .btn-danger:hover {
                background: #dd0000;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .result {
                display: none;
                margin-top: 30px;
                text-align: center;
            }
            
            .signal-box {
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 24px;
                font-weight: bold;
                color: white;
            }
            
            .signal-red {
                background: #ff4444;
            }
            
            .signal-yellow {
                background: #ffaa00;
                color: #333;
            }
            
            .signal-green {
                background: #00cc44;
            }
            
            .signal-none {
                background: #999;
            }
            
            .result-image {
                max-width: 100%;
                border-radius: 8px;
                margin-bottom: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            
            .error-message {
                display: none;
                background: #ffebee;
                color: #c62828;
                padding: 15px;
                border-radius: 6px;
                margin-top: 15px;
                border-left: 4px solid #c62828;
            }
            
            .status-text {
                color: #666;
                font-size: 14px;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚦 Traffic Signal Recognition</h1>
            <p class="subtitle">Upload an image or use your webcam to detect traffic lights</p>
            
            <div class="tabs">
                <button class="tab-button active" onclick="switchTab(\'webcam\')">📷 Webcam</button>
                <button class="tab-button" onclick="switchTab(\'upload\')">📸 Image Upload</button>
            </div>
            
            <!-- Webcam Tab -->
            <div id="webcam" class="tab-content active">
                <div class="webcam-container">
                    <video id="video" autoplay playsinline style="display: none;"></video>
                    <canvas id="canvas"></canvas>
                    <div id="webcamPlaceholder" style="background: #f0f0f0; border-radius: 8px; padding: 60px 20px; margin-bottom: 20px;">
                        <div style="font-size: 48px; margin-bottom: 10px;">📹</div>
                        <div style="color: #999; font-size: 16px;">Camera will appear here</div>
                    </div>
                    
                    <div class="button-group">
                        <button class="btn-primary" onclick="startWebcam()" id="startBtn">Start Webcam</button>
                        <button class="btn-danger" onclick="stopWebcam()" id="stopBtn" style="display: none;">Stop Webcam</button>
                        <button class="btn-secondary" onclick="captureFrame()" id="captureBtn" style="display: none;">Capture & Detect</button>
                    </div>
                    
                    <p class="status-text" id="webcamStatus"></p>
                </div>
                
                <div class="loading" id="webcamLoading">
                    <div class="spinner"></div>
                    <p>Processing frame...</p>
                </div>
                
                <div class="error-message" id="webcamError"></div>
                
                <div class="result" id="webcamResult">
                    <img id="webcamResultImage" class="result-image" src="" alt="Result">
                    <div class="signal-box" id="webcamSignalBox"></div>
                </div>
            </div>
            
            <!-- Image Upload Tab -->
            <div id="upload" class="tab-content">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📸</div>
                    <div class="upload-text">Click or drag image here</div>
                    <div class="upload-subtext">Supports PNG, JPG, JPEG, GIF, BMP (max 16MB)</div>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
                
                <div class="button-group">
                    <button class="btn-primary" onclick="document.getElementById(\'fileInput\').click()">
                        Choose Image
                    </button>
                    <button class="btn-secondary" onclick="resetImageForm()" id="imageResetBtn" style="display: none;">
                        Clear
                    </button>
                </div>
                
                <div class="loading" id="imageLoading">
                    <div class="spinner"></div>
                    <p>Processing image...</p>
                </div>
                
                <div class="error-message" id="imageError"></div>
                
                <div class="result" id="imageResult">
                    <img id="imageResultImage" class="result-image" src="" alt="Result">
                    <div class="signal-box" id="imageSignalBox"></div>
                    <button class="btn-secondary" onclick="resetImageForm()">Detect Another</button>
                </div>
            </div>
        </div>
        
        <script>
            const signalStyles = {
                'red': 'signal-red',
                'yellow': 'signal-yellow',
                'green': 'signal-green',
                'none': 'signal-none'
            };
            
            // Tab switching
            function switchTab(tab) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-button').forEach(el => el.classList.remove('active'));
                document.getElementById(tab).classList.add('active');
                event.target.classList.add('active');
            }
            
            // ============ WEBCAM FUNCTIONS ============
            let stream = null;
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            
            async function startWebcam() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'environment' }
                    });
                    
                    video.srcObject = stream;
                    video.style.display = 'block';
                    document.getElementById('webcamPlaceholder').style.display = 'none';
                    
                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('stopBtn').style.display = 'inline-block';
                    document.getElementById('captureBtn').style.display = 'inline-block';
                    document.getElementById('webcamStatus').textContent = '✅ Camera ready! Click "Capture & Detect" to analyze the traffic light.';
                    
                    hideError('webcam');
                } catch (error) {
                    showWebcamError('Camera access denied. Please allow camera permission.');
                }
            }
            
            function stopWebcam() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                
                video.style.display = 'none';
                document.getElementById('webcamPlaceholder').style.display = 'block';
                
                document.getElementById('startBtn').style.display = 'inline-block';
                document.getElementById('stopBtn').style.display = 'none';
                document.getElementById('captureBtn').style.display = 'none';
                document.getElementById('webcamStatus').textContent = '';
                document.getElementById('webcamResult').style.display = 'none';
            }
            
            function captureFrame() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                canvas.toBlob(blob => {
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');
                    
                    showLoading('webcam');
                    hideError('webcam');
                    
                    fetch('/api/detect', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        hideLoading('webcam');
                        if (data.success) {
                            showWebcamResult(data);
                        } else {
                            showWebcamError(data.error || 'Detection failed');
                        }
                    })
                    .catch(error => {
                        hideLoading('webcam');
                        showWebcamError('Error: ' + error.message);
                    });
                }, 'image/jpeg', 0.8);
            }
            
            // ============ IMAGE UPLOAD FUNCTIONS ============
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const imageLoading = document.getElementById('imageLoading');
            const imageResult = document.getElementById('imageResult');
            const imageError = document.getElementById('imageError');
            
            uploadArea.addEventListener('click', () => fileInput.click());
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length) {
                    fileInput.files = files;
                    handleFileSelect();
                }
            });
            
            fileInput.addEventListener('change', handleFileSelect);
            
            function handleFileSelect() {
                const file = fileInput.files[0];
                if (!file) return;
                
                if (!file.type.startsWith('image/')) {
                    showImageError('Please select a valid image file');
                    return;
                }
                
                uploadImage(file);
            }
            
            function uploadImage(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                showLoading('image');
                hideError('image');
                
                fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    hideLoading('image');
                    if (data.success) {
                        showImageResult(data);
                    } else {
                        showImageError(data.error || 'Detection failed');
                    }
                })
                .catch(error => {
                    hideLoading('image');
                    showImageError('Error: ' + error.message);
                });
            }
            
            // ============ UI HELPERS ============
            function showLoading(type) {
                document.getElementById(type + 'Loading').style.display = 'block';
            }
            
            function hideLoading(type) {
                document.getElementById(type + 'Loading').style.display = 'none';
            }
            
            function showWebcamResult(data) {
                document.getElementById('webcamResultImage').src = data.image;
                document.getElementById('webcamSignalBox').textContent = data.signal_text;
                document.getElementById('webcamSignalBox').className = 'signal-box ' + signalStyles[data.signal];
                document.getElementById('webcamResult').style.display = 'block';
                document.getElementById('webcamStatus').textContent = '✅ ' + data.signal_text;
            }
            
            function showImageResult(data) {
                document.getElementById('imageResultImage').src = data.image;
                document.getElementById('imageSignalBox').textContent = data.signal_text;
                document.getElementById('imageSignalBox').className = 'signal-box ' + signalStyles[data.signal];
                document.getElementById('imageResult').style.display = 'block';
                document.getElementById('imageResetBtn').style.display = 'inline-block';
            }
            
            function showWebcamError(message) {
                const errorEl = document.getElementById('webcamError');
                errorEl.textContent = '❌ ' + message;
                errorEl.style.display = 'block';
            }
            
            function showImageError(message) {
                const errorEl = document.getElementById('imageError');
                errorEl.textContent = '❌ ' + message;
                errorEl.style.display = 'block';
            }
            
            function hideError(type) {
                document.getElementById(type + 'Error').style.display = 'none';
            }
            
            function resetImageForm() {
                fileInput.value = '';
                uploadArea.style.display = 'block';
                document.getElementById('imageResult').style.display = 'none';
                document.getElementById('imageError').style.display = 'none';
                document.getElementById('imageResetBtn').style.display = 'none';
            }
        </script>
    </body>
    </html>
    ''', 200

if __name__ == '__main__':
    app.run(debug=True)
