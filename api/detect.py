from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

# Import detector from parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    """Serve the web interface."""
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
                max-width: 700px;
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
            
            .button-group {
                display: flex;
                gap: 10px;
                margin-top: 20px;
                justify-content: center;
            }
            
            button {
                padding: 10px 20px;
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
            
            .btn-secondary {
                background: #f0f0f0;
                color: #333;
            }
            
            .btn-secondary:hover {
                background: #e0e0e0;
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚦 Traffic Signal Recognition</h1>
            <p class="subtitle">Upload an image to detect traffic light colors</p>
            
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📸</div>
                <div class="upload-text">Click or drag image here</div>
                <div class="upload-subtext">Supports PNG, JPG, JPEG, GIF, BMP (max 16MB)</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="button-group">
                <button class="btn-primary" onclick="document.getElementById('fileInput').click()">
                    Choose Image
                </button>
                <button class="btn-secondary" onclick="resetForm()" id="resetBtn" style="display: none;">
                    Clear
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing image...</p>
            </div>
            
            <div class="error-message" id="errorMessage"></div>
            
            <div class="result" id="result">
                <img id="resultImage" class="result-image" src="" alt="Result">
                <div class="signal-box" id="signalBox"></div>
                <button class="btn-secondary" onclick="resetForm()">Detect Another</button>
            </div>
        </div>
        
        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const resultImage = document.getElementById('resultImage');
            const signalBox = document.getElementById('signalBox');
            const errorMessage = document.getElementById('errorMessage');
            const resetBtn = document.getElementById('resetBtn');
            
            const signalStyles = {
                'red': 'signal-red',
                'yellow': 'signal-yellow',
                'green': 'signal-green',
                'none': 'signal-none'
            };
            
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
                    showError('Please select a valid image file');
                    return;
                }
                
                uploadImage(file);
            }
            
            function uploadImage(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                showLoading();
                
                fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showResult(data);
                    } else {
                        showError(data.error || 'Detection failed');
                    }
                })
                .catch(error => {
                    showError('Error: ' + error.message);
                })
                .finally(() => {
                    hideLoading();
                });
            }
            
            function showLoading() {
                uploadArea.style.display = 'none';
                result.style.display = 'none';
                errorMessage.style.display = 'none';
                resetBtn.style.display = 'none';
                loading.style.display = 'block';
            }
            
            function hideLoading() {
                loading.style.display = 'none';
            }
            
            function showResult(data) {
                resultImage.src = data.image;
                signalBox.textContent = data.signal_text;
                signalBox.className = 'signal-box ' + signalStyles[data.signal];
                result.style.display = 'block';
                resetBtn.style.display = 'inline-block';
            }
            
            function showError(message) {
                errorMessage.textContent = '❌ ' + message;
                errorMessage.style.display = 'block';
                uploadArea.style.display = 'block';
            }
            
            function resetForm() {
                fileInput.value = '';
                uploadArea.style.display = 'block';
                result.style.display = 'none';
                errorMessage.style.display = 'none';
                resetBtn.style.display = 'none';
            }
        </script>
    </body>
    </html>
    ''', 200

if __name__ == '__main__':
    app.run(debug=True)
