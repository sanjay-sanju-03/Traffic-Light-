from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

# Import detectors from src directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from unified_detector import UnifiedTrafficDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize unified detector
try:
    detector = UnifiedTrafficDetector(enable_lights=True, enable_signs=True)
except Exception as e:
    print(f"Warning: {e}")
    detector = UnifiedTrafficDetector(enable_lights=True, enable_signs=False)

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
    API endpoint for complete traffic detection (lights + signs).
    Accepts image file or base64 encoded image.
    Returns detected lights, signs, and annotated image.
    """
    try:
        if 'file' not in request.files and 'image' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        image = None

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
            file_stream = file.read()
            nparr = np.frombuffer(file_stream, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

        height, width = image.shape[:2]
        if width > 800 or height > 600:
            scale = min(800/width, 600/height)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        result = detector.detect_all(image)

        light_info = result.get('lights', {})
        sign_info  = result.get('signs', {})
        annotated  = result.get('annotated_frame', image)

        result_image_b64 = image_to_base64(annotated)

        response_data = {
            'success': True,
            'traffic_light': {
                'detected':   light_info.get('signal', 'unknown') if light_info else 'unknown',
                'text':       light_info.get('text', 'No light') if light_info else 'No light',
                'color_hex':  '#{:02x}{:02x}{:02x}'.format(
                    light_info['color'][2], light_info['color'][1], light_info['color'][0]
                ) if light_info and light_info.get('color') else '#ffffff'
            },
            'traffic_signs': {
                'count':  len(sign_info.get('signs', [])) if sign_info else 0,
                'signs':  sign_info.get('signs', []) if sign_info else [],
                'status': sign_info.get('status', 'No signs') if sign_info else 'No signs'
            },
            'image': result_image_b64
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'Traffic Detection System',
        'features': {
            'traffic_lights': 'enabled (RED/YELLOW/GREEN)',
            'traffic_signs':  'enabled (STOP + more)',
            'realtime_webcam': 'supported',
            'image_upload':   'supported',
        }
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Serve the real-time traffic detection web interface."""
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Detection System – Live</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background: #0d0d1a;
            color: #c9c9e0;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* ── Header ── */
        header {
            flex-shrink: 0;
            background: #11112a;
            border-bottom: 1px solid #1e1e40;
            padding: 0 24px;
            height: 58px;
            display: flex;
            align-items: center;
            gap: 12px;
            z-index: 10;
        }
        .logo { font-size: 26px; }
        .app-title {
            font-size: 18px;
            font-weight: 800;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .live-pill {
            margin-left: auto;
            display: none;
            align-items: center;
            gap: 6px;
            background: #ef444422;
            border: 1px solid #ef444466;
            color: #ef4444;
            border-radius: 999px;
            padding: 4px 14px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        .live-pill.visible { display: flex; }
        .live-dot {
            width: 7px; height: 7px;
            background: #ef4444;
            border-radius: 50%;
            animation: blink 1.2s ease-in-out infinite;
        }
        @keyframes blink {
            0%,100% { opacity:1; } 50% { opacity:.2; }
        }

        /* ── Body layout ── */
        .body-layout {
            flex: 1;
            display: flex;
            overflow: hidden;
        }

        /* ── Main canvas area ── */
        .canvas-wrap {
            flex: 1;
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        #liveCanvas {
            max-width: 100%;
            max-height: 100%;
            display: block;
            object-fit: contain;
        }
        #video, #capCanvas { display: none; }

        .empty-state {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
            color: #2a2a50;
            pointer-events: none;
        }
        .empty-state .e-icon { font-size: 64px; }
        .empty-state p { font-size: 15px; }

        .fps-chip {
            position: absolute;
            top: 14px;
            right: 14px;
            background: rgba(0,0,0,.75);
            border: 1px solid #2a2a50;
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 12px;
            font-weight: 700;
            color: #818cf8;
            display: none;
        }

        /* ── Sidebar ── */
        .sidebar {
            width: 320px;
            flex-shrink: 0;
            background: #11112a;
            border-left: 1px solid #1e1e40;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Tab bar */
        .tab-bar {
            display: flex;
            border-bottom: 1px solid #1e1e40;
            flex-shrink: 0;
        }
        .tab-btn {
            flex: 1;
            padding: 13px 8px;
            background: none;
            border: none;
            color: #44446a;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all .2s;
        }
        .tab-btn.active { color: #818cf8; border-bottom-color: #818cf8; }

        /* Scroll container for panels */
        .panel { flex: 1; overflow-y: auto; display: none; flex-direction: column; }
        .panel.active { display: flex; }

        /* Sections */
        .sec {
            padding: 18px 18px 14px;
            border-bottom: 1px solid #1a1a38;
        }
        .sec-title {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #3a3a60;
            margin-bottom: 12px;
        }

        /* Traffic Light Widget */
        .tl-widget {
            background: #16163c;
            border: 1px solid #20204a;
            border-radius: 14px;
            padding: 18px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .tl-pole {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            background: #0d0d22;
            border: 1px solid #20204a;
            border-radius: 10px;
            padding: 10px 8px;
        }
        .tl-bulb {
            width: 28px; height: 28px;
            border-radius: 50%;
            background: #1a1a35;
            border: 2px solid #22224a;
            transition: all .3s ease;
        }
        .tl-bulb.lit-red    { background:#ef4444; border-color:#fca5a5; box-shadow:0 0 16px #ef444499, 0 0 32px #ef444444; }
        .tl-bulb.lit-yellow { background:#f59e0b; border-color:#fde68a; box-shadow:0 0 16px #f59e0b99, 0 0 32px #f59e0b44; }
        .tl-bulb.lit-green  { background:#22c55e; border-color:#86efac; box-shadow:0 0 16px #22c55e99, 0 0 32px #22c55e44; }
        .tl-info { flex: 1; }
        .tl-state {
            font-size: 22px;
            font-weight: 800;
            color: #e0e0f0;
            line-height: 1.1;
        }
        .tl-state span { display: block; font-size: 11px; font-weight: 500; color: #44446a; margin-top: 3px; }

        /* Signs */
        .signs-container { display: flex; flex-direction: column; gap: 7px; }
        .sign-chip {
            display: flex;
            align-items: center;
            gap: 9px;
            background: #16163c;
            border: 1px solid #20204a;
            border-radius: 8px;
            padding: 9px 12px;
            font-size: 13px;
            font-weight: 600;
            animation: fadeSlide .25s ease;
        }
        @keyframes fadeSlide {
            from { opacity:0; transform:translateX(8px); }
            to   { opacity:1; transform:translateX(0); }
        }
        .sign-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
        .no-sign { font-size:13px; color:#2e2e55; padding: 6px 0; }

        /* Stats */
        .stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
        .stat-box {
            background: #16163c;
            border: 1px solid #20204a;
            border-radius: 10px;
            padding: 11px 13px;
        }
        .stat-lbl { font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:#33335a; }
        .stat-val { font-size:20px; font-weight:800; color:#818cf8; margin-top:3px; line-height:1; }
        .stat-unit { font-size:10px; color:#444466; }

        /* Interval slider */
        .slider-row {
            display:flex; align-items:center; gap:8px; margin-top:12px;
        }
        .slider-row label { font-size:11px; color:#44446a; white-space:nowrap; }
        input[type=range] { flex:1; accent-color:#818cf8; cursor:pointer; }
        .slider-val { font-size:11px; font-weight:700; color:#818cf8; min-width:38px; text-align:right; }

        /* Buttons */
        .btn-stack { display:flex; flex-direction:column; gap:8px; }
        .btn {
            width:100%; padding:12px;
            border:none; border-radius:10px;
            font-family:'Inter',sans-serif; font-size:14px; font-weight:700;
            cursor:pointer; transition:all .2s;
            display:flex; align-items:center; justify-content:center; gap:8px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #fff;
        }
        .btn-primary:hover:not(:disabled) { transform:translateY(-1px); box-shadow:0 8px 20px #6366f144; }
        .btn-warn {
            background: #1a1a38;
            border: 1px solid #f59e0b55;
            color: #f59e0b;
        }
        .btn-warn:hover:not(:disabled) { background:#f59e0b18; }
        .btn-danger {
            background: #1a1a38;
            border: 1px solid #ef444455;
            color: #ef4444;
        }
        .btn-danger:hover:not(:disabled) { background:#ef444418; }
        .btn:disabled { opacity:.3; cursor:not-allowed; }

        /* Upload */
        .upload-zone {
            border: 2px dashed #20204a;
            border-radius: 10px;
            padding: 28px 16px;
            text-align: center;
            cursor: pointer;
            transition: all .2s;
            background: #16163c;
        }
        .upload-zone:hover { border-color:#818cf8; background:#18184a; }
        .upload-zone.over  { border-color:#c084fc; background:#1e1650; }
        .upload-zone .u-icon { font-size:30px; margin-bottom:8px; }
        .upload-zone .u-txt  { font-size:12px; color:#44446a; line-height:1.6; }
        input[type=file] { display:none; }

        /* Error */
        .err-box {
            background:#1a0a14; border:1px solid #ef444433;
            color:#ef4444; border-radius:8px; padding:9px 12px;
            font-size:12px; margin-top:10px; display:none;
        }

        /* Upload result */
        .u-result { display:none; }
        .u-tl-widget {
            background: #16163c;
            border:1px solid #20204a;
            border-radius:14px;
            padding:14px;
            display:flex; align-items:center; gap:14px;
            margin-bottom:12px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width:3px; }
        ::-webkit-scrollbar-track { background:#0d0d1a; }
        ::-webkit-scrollbar-thumb { background:#20204a; border-radius:4px; }
    </style>
</head>
<body>

<header>
    <span class="logo">🚦</span>
    <span class="app-title">Traffic Detection System</span>
    <div class="live-pill" id="livePill">
        <div class="live-dot"></div> LIVE
    </div>
</header>

<div class="body-layout">

    <!-- ── Video ── -->
    <div class="canvas-wrap">
        <div class="empty-state" id="emptyState">
            <div class="e-icon">📹</div>
            <p>Start Live Detection to begin</p>
        </div>
        <canvas id="liveCanvas"></canvas>
        <video id="video" autoplay playsinline muted></video>
        <canvas id="capCanvas"></canvas>
        <div class="fps-chip" id="fpsChip">— FPS</div>
    </div>

    <!-- ── Sidebar ── -->
    <div class="sidebar">

        <!-- Tab Bar -->
        <div class="tab-bar">
            <button class="tab-btn active" id="tabLive"   onclick="switchTab('live')">📷 Live</button>
            <button class="tab-btn"        id="tabUpload" onclick="switchTab('upload')">📸 Upload</button>
        </div>

        <!-- ══ LIVE PANEL ══ -->
        <div class="panel active" id="panelLive">

            <!-- Traffic Light -->
            <div class="sec">
                <div class="sec-title">Traffic Light</div>
                <div class="tl-widget">
                    <div class="tl-pole">
                        <div class="tl-bulb" id="bRed"></div>
                        <div class="tl-bulb" id="bYellow"></div>
                        <div class="tl-bulb" id="bGreen"></div>
                    </div>
                    <div class="tl-info">
                        <div class="tl-state" id="tlState">
                            —
                            <span id="tlSub">Waiting...</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Signs -->
            <div class="sec">
                <div class="sec-title">Detected Signs</div>
                <div class="signs-container" id="signsCont">
                    <p class="no-sign">No signs detected</p>
                </div>
            </div>

            <!-- Stats -->
            <div class="sec">
                <div class="sec-title">Performance</div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-lbl">Frames</div>
                        <div class="stat-val" id="sFrames">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-lbl">Latency</div>
                        <div class="stat-val" id="sLatency">—</div>
                        <span class="stat-unit">ms</span>
                    </div>
                    <div class="stat-box">
                        <div class="stat-lbl">FPS</div>
                        <div class="stat-val" id="sFps">—</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-lbl">Signs</div>
                        <div class="stat-val" id="sSigns">0</div>
                    </div>
                </div>
                <div class="slider-row">
                    <label>Interval</label>
                    <input type="range" id="iSlider" min="200" max="2000" step="100" value="500"
                           oninput="setInterval2(this.value)">
                    <span class="slider-val" id="iVal">500ms</span>
                </div>
            </div>

            <!-- Controls -->
            <div class="sec">
                <div class="sec-title">Controls</div>
                <div class="btn-stack">
                    <button class="btn btn-primary" id="btnStart" onclick="startLive()">▶ &nbsp;Start Live Detection</button>
                    <button class="btn btn-warn"    id="btnPause" onclick="togglePause()" disabled>⏸ &nbsp;Pause</button>
                    <button class="btn btn-danger"  id="btnStop"  onclick="stopLive()"   disabled>⏹ &nbsp;Stop</button>
                </div>
                <div class="err-box" id="liveErr"></div>
            </div>
        </div>

        <!-- ══ UPLOAD PANEL ══ -->
        <div class="panel" id="panelUpload">
            <div class="sec">
                <div class="sec-title">Upload Image</div>
                <div class="upload-zone" id="uploadZone">
                    <div class="u-icon">📸</div>
                    <div class="u-txt" id="uTxt">Click or drag &amp; drop an image<br>
                        <small style="color:#2e2e55">PNG · JPG · BMP  (max 16 MB)</small>
                    </div>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
                <div class="err-box" id="uploadErr"></div>
            </div>

            <div class="sec u-result" id="uResultSec">
                <div class="sec-title">Result</div>
                <div class="u-tl-widget">
                    <div class="tl-pole">
                        <div class="tl-bulb" id="uBRed"></div>
                        <div class="tl-bulb" id="uBYellow"></div>
                        <div class="tl-bulb" id="uBGreen"></div>
                    </div>
                    <div class="tl-info">
                        <div class="tl-state" id="uTlState">—<span id="uTlSub"></span></div>
                    </div>
                </div>
                <div class="signs-container" id="uSignsCont"></div>
            </div>
        </div>

    </div><!-- /sidebar -->
</div><!-- /body-layout -->

<script>
/* =====================================================
   STATE
===================================================== */
let camStream   = null;
let detectTimer = null;
let fpsTimer2   = null;
let isLive      = false;
let paused      = false;
let frames      = 0;
let fpsBucket   = 0;
let fps         = 0;
let intervalMs  = 500;
let requesting  = false;

const video      = document.getElementById('video');
const capCanvas  = document.getElementById('capCanvas');
const liveCanvas = document.getElementById('liveCanvas');
const liveCtx    = liveCanvas.getContext('2d');
const capCtx     = capCanvas.getContext('2d');

/* =====================================================
   TAB SWITCHING
===================================================== */
function switchTab(tab) {
    ['live','upload'].forEach(t => {
        document.getElementById('panel' + t.charAt(0).toUpperCase() + t.slice(1)).classList.toggle('active', t === tab);
        document.getElementById('tab'   + t.charAt(0).toUpperCase() + t.slice(1)).classList.toggle('active', t === tab);
    });
}

/* =====================================================
   LIVE DETECTION
===================================================== */
async function startLive() {
    clearErr('live');
    try {
        camStream = await navigator.mediaDevices.getUserMedia({
            video: { width:{ideal:640}, height:{ideal:480}, facingMode:'user' }
        });
        video.srcObject = camStream;
        await new Promise(r => { video.onloadedmetadata = r; });
        video.play();

        liveCanvas.width  = video.videoWidth  || 640;
        liveCanvas.height = video.videoHeight || 480;
        capCanvas.width   = liveCanvas.width;
        capCanvas.height  = liveCanvas.height;

        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('livePill').classList.add('visible');
        document.getElementById('fpsChip').style.display = 'block';

        setButtons(false, true, true);
        isLive = true; paused = false;

        renderLoop();
        detectTimer = setInterval(sendFrame, intervalMs);
        fpsTimer2   = setInterval(() => {
            fps = fpsBucket; fpsBucket = 0;
            document.getElementById('fpsChip').textContent = fps + ' FPS';
            document.getElementById('sFps').textContent    = fps;
        }, 1000);

    } catch(e) {
        showErr('live', 'Camera access denied — please allow permission.');
    }
}

function renderLoop() {
    if (!isLive) return;
    if (video.readyState >= 2) {
        liveCtx.drawImage(video, 0, 0, liveCanvas.width, liveCanvas.height);
        fpsBucket++;
    }
    requestAnimationFrame(renderLoop);
}

function sendFrame() {
    if (!isLive || paused || requesting) return;
    if (video.readyState < 2) return;

    capCtx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
    const t0 = performance.now();
    requesting = true;

    capCanvas.toBlob(blob => {
        if (!blob) { requesting = false; return; }
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');

        fetch('/api/detect', { method:'POST', body:fd })
        .then(r => r.json())
        .then(data => {
            requesting = false;
            const lat = Math.round(performance.now() - t0);
            frames++;
            document.getElementById('sFrames').textContent  = frames;
            document.getElementById('sLatency').textContent = lat;

            if (data.success) {
                updateLight(data.traffic_light, 'b', 'tl');
                updateSigns(data.traffic_signs, 'signsCont', 'sSigns');

                if (data.image) {
                    const img = new Image();
                    img.onload = () => liveCtx.drawImage(img, 0, 0, liveCanvas.width, liveCanvas.height);
                    img.src = data.image;
                }
            }
        })
        .catch(() => { requesting = false; });
    }, 'image/jpeg', 0.75);
}

function togglePause() {
    paused = !paused;
    document.getElementById('btnPause').textContent = paused ? '▶  Resume' : '⏸  Pause';
}

function stopLive() {
    isLive = false;
    requesting = false;
    clearInterval(detectTimer);
    clearInterval(fpsTimer2);
    if (camStream) camStream.getTracks().forEach(t => t.stop());
    camStream = null;

    liveCtx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);
    document.getElementById('emptyState').style.display = 'flex';
    document.getElementById('livePill').classList.remove('visible');
    document.getElementById('fpsChip').style.display = 'none';

    setButtons(true, false, false);
    frames = 0;
    document.getElementById('sFrames').textContent  = 0;
    document.getElementById('sLatency').textContent = '—';
    document.getElementById('sFps').textContent     = '—';
    document.getElementById('sSigns').textContent   = 0;
    resetLight('b', 'tl');
    document.getElementById('signsCont').innerHTML = '<p class="no-sign">No signs detected</p>';
}

function setInterval2(v) {
    intervalMs = parseInt(v);
    document.getElementById('iVal').textContent = v + 'ms';
    if (isLive) {
        clearInterval(detectTimer);
        detectTimer = setInterval(sendFrame, intervalMs);
    }
}

/* =====================================================
   UI: TRAFFIC LIGHT
===================================================== */
function updateLight(light, bulbPfx, statePfx) {
    const sig  = (light && light.detected) ? light.detected.toLowerCase() : 'none';
    const txt  = (light && light.text)     ? light.text : '—';
    ['Red','Yellow','Green'].forEach(c => {
        const el = document.getElementById(bulbPfx + c);
        if (el) el.className = 'tl-bulb' + (sig === c.toLowerCase() ? ' lit-' + c.toLowerCase() : '');
    });
    const stateEl = document.getElementById(statePfx + 'State');
    const subEl   = document.getElementById(statePfx + 'Sub');
    if (stateEl) stateEl.childNodes[0].textContent = txt + '\n';
    if (subEl)   subEl.textContent = sig === 'none' ? 'No signal detected' : 'Signal confirmed ✓';
}

function resetLight(bulbPfx, statePfx) {
    ['Red','Yellow','Green'].forEach(c => {
        const el = document.getElementById(bulbPfx + c);
        if (el) el.className = 'tl-bulb';
    });
    const stateEl = document.getElementById(statePfx + 'State');
    const subEl   = document.getElementById(statePfx + 'Sub');
    if (stateEl) stateEl.childNodes[0].textContent = '—\n';
    if (subEl)   subEl.textContent = 'Waiting...';
}

/* =====================================================
   UI: SIGNS
===================================================== */
const SCOLORS = { 'stop':'#ef4444','yield':'#f59e0b','speed':'#3b82f6','park':'#8b5cf6','no entry':'#ef4444' };
function sColor(n) {
    n = n.toLowerCase();
    for (const [k,v] of Object.entries(SCOLORS)) if (n.includes(k)) return v;
    return '#22c55e';
}

function updateSigns(sigData, contId, cntId) {
    const cont  = document.getElementById(contId);
    const signs = (sigData && sigData.signs) ? sigData.signs : [];
    if (cntId) document.getElementById(cntId).textContent = signs.length;

    if (!signs.length) {
        cont.innerHTML = '<p class="no-sign">No signs detected</p>';
        return;
    }
    const key = signs.join(',');
    if (cont.dataset.key === key) return;
    cont.dataset.key = key;
    cont.innerHTML = signs.map(s =>
        `<div class="sign-chip">
            <div class="sign-dot" style="background:${sColor(s)}"></div>
            <span>${s}</span>
         </div>`
    ).join('');
}

/* =====================================================
   UPLOAD
===================================================== */
const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('over'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('over');
    if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; doUpload(); }
});
fileInput.addEventListener('change', doUpload);

function doUpload() {
    const file = fileInput.files[0];
    if (!file || !file.type.startsWith('image/')) { showErr('upload','Please select a valid image.'); return; }
    clearErr('upload');
    const uTxt = document.getElementById('uTxt');
    uTxt.textContent = '⏳ Processing...';

    const fd = new FormData();
    fd.append('file', file);

    fetch('/api/detect', { method:'POST', body:fd })
    .then(r => r.json())
    .then(data => {
        uTxt.innerHTML = 'Click or drag &amp; drop an image<br><small style="color:#2e2e55">PNG · JPG · BMP  (max 16 MB)</small>';
        if (!data.success) { showErr('upload', data.error || 'Detection failed'); return; }

        if (data.image) {
            const img = new Image();
            img.onload = () => {
                document.getElementById('emptyState').style.display = 'none';
                liveCanvas.width  = img.naturalWidth;
                liveCanvas.height = img.naturalHeight;
                liveCtx.drawImage(img, 0, 0);
            };
            img.src = data.image;
        }

        document.getElementById('uResultSec').style.display = '';
        updateLight(data.traffic_light, 'uB', 'uTl');
        updateSigns(data.traffic_signs, 'uSignsCont', null);
    })
    .catch(e => {
        uTxt.innerHTML = 'Click or drag &amp; drop an image';
        showErr('upload','Request failed: ' + e.message);
    });
}

/* =====================================================
   MISC
===================================================== */
function setButtons(start, pause, stop) {
    document.getElementById('btnStart').disabled = !start;
    document.getElementById('btnPause').disabled = !pause;
    document.getElementById('btnStop').disabled  = !stop;
}
function showErr(ctx, msg) {
    const el = document.getElementById(ctx === 'live' ? 'liveErr' : 'uploadErr');
    el.textContent = '❌ ' + msg; el.style.display = 'block';
}
function clearErr(ctx) {
    const el = document.getElementById(ctx === 'live' ? 'liveErr' : 'uploadErr');
    el.style.display = 'none';
}
</script>
</body>
</html>"""
    return html, 200


if __name__ == '__main__':
    app.run(debug=True)
