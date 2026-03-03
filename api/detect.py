"""
Traffic Detection API – Vercel-compatible (HSV-based, no PyTorch/YOLO).

Traffic light detection  → HSV colour masking  (runs everywhere)
Traffic sign detection   → HSV shape analysis  (runs everywhere)
YOLOv8 sign detection   → local only           (too large for Vercel)
"""
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os, sys, base64

# ── Path setup ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

# ── Lightweight HSV detector (always available) ─────────────────
from signal_detector import TrafficDetector
hsv_detector = TrafficDetector()

# ── Optional: try to load the full YOLO-based unified detector ──
# (works locally; silently skipped on Vercel / resource-limited envs)
try:
    from unified_detector import UnifiedTrafficDetector
    full_detector = UnifiedTrafficDetector(enable_lights=True, enable_signs=True)
    FULL_DETECTOR = True
except Exception as _e:
    FULL_DETECTOR = False

# ── Flask app ────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB

ALLOWED = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

def to_b64(cv_img):
    _, buf = cv2.imencode('.jpg', cv_img)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

def detect_hsv(image):
    """HSV-based detection: traffic lights + shape-based signs."""
    # ── Traffic light ────────────────────────────────────────────
    sig_key, sig_text, sig_color = hsv_detector.detect_light(image)

    # Annotate: coloured bar + label
    annotated = image.copy()
    cv2.rectangle(annotated, (10, 10), (220, 60), sig_color, -1)
    cv2.putText(annotated, sig_text, (20, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ── Signs ────────────────────────────────────────────────────
    sign_detections = hsv_detector.detect_signs(image)
    signs_found = []
    for det in sign_detections:
        if det.get('type') == 'none':
            continue
        name = det.get('name', '')
        signs_found.append(name)
        if 'box' in det:
            x, y, w, h = det['box']
            color = det.get('color', (0, 255, 0))
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated, name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return {
        'success': True,
        'traffic_light': {
            'detected':  sig_key,
            'text':      sig_text,
            'color_hex': '#{:02x}{:02x}{:02x}'.format(
                sig_color[2], sig_color[1], sig_color[0])
        },
        'traffic_signs': {
            'count':  len(signs_found),
            'signs':  signs_found,
            'status': f'Detected {len(signs_found)} sign(s)' if signs_found else 'No signs detected'
        },
        'image': to_b64(annotated),
        'mode': 'hsv'
    }

def detect_full(image):
    """Full YOLO-based detection (local only)."""
    result = full_detector.detect_all(image)
    light  = result.get('lights', {})
    signs  = result.get('signs', {})
    ann    = result.get('annotated_frame', image)
    return {
        'success': True,
        'traffic_light': {
            'detected':  light.get('signal', 'unknown') if light else 'unknown',
            'text':      light.get('text', 'No light')  if light else 'No light',
            'color_hex': '#{:02x}{:02x}{:02x}'.format(
                light['color'][2], light['color'][1], light['color'][0]
            ) if light and light.get('color') else '#ffffff'
        },
        'traffic_signs': {
            'count':  len(signs.get('signs', [])) if signs else 0,
            'signs':  signs.get('signs', [])      if signs else [],
            'status': signs.get('status', 'No signs') if signs else 'No signs'
        },
        'image': to_b64(ann),
        'mode': 'yolo'
    }

def read_image_from_request():
    """Return decoded OpenCV image from uploaded file or base64 form field."""
    if 'file' in request.files:
        f = request.files['file']
        if f.filename == '' or not allowed(f.filename):
            return None, 'Invalid or missing file'
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR), None

    if 'image' in request.form:
        try:
            b64 = request.form['image']
            if b64.startswith('data:image'):
                b64 = b64.split(',')[1]
            data = np.frombuffer(base64.b64decode(b64), np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR), None
        except Exception as e:
            return None, str(e)

    return None, 'No image provided'


# ── Routes ───────────────────────────────────────────────────────

@app.route('/api/detect', methods=['POST'])
def detect_signal():
    try:
        if 'file' not in request.files and 'image' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        image, err = read_image_from_request()
        if image is None:
            return jsonify({'error': err or 'Failed to decode image'}), 400

        # Downscale if needed
        h, w = image.shape[:2]
        if w > 800 or h > 600:
            scale = min(800/w, 600/h)
            image = cv2.resize(image, (int(w*scale), int(h*scale)))

        # Use best available detector
        result = detect_full(image) if FULL_DETECTOR else detect_hsv(image)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':   'ok',
        'service':  'Traffic Detection System',
        'detector': 'yolo + hsv' if FULL_DETECTOR else 'hsv (lightweight)',
        'features': {
            'traffic_lights':  'enabled (RED / YELLOW / GREEN)',
            'traffic_signs':   'enabled',
            'realtime_webcam': 'supported',
            'image_upload':    'supported',
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
            background: #0d0d1a; color: #c9c9e0;
            height: 100vh; overflow: hidden;
            display: flex; flex-direction: column;
        }
        header {
            flex-shrink: 0; background: #11112a;
            border-bottom: 1px solid #1e1e40;
            padding: 0 24px; height: 58px;
            display: flex; align-items: center; gap: 12px;
        }
        .logo { font-size: 26px; }
        .app-title {
            font-size: 18px; font-weight: 800;
            background: linear-gradient(90deg,#818cf8,#c084fc);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .live-pill {
            margin-left: auto; display: none;
            align-items: center; gap: 6px;
            background: #ef444422; border: 1px solid #ef444466;
            color: #ef4444; border-radius: 999px;
            padding: 4px 14px; font-size: 11px; font-weight: 700; letter-spacing: 1px;
        }
        .live-pill.visible { display: flex; }
        .live-dot { width:7px; height:7px; background:#ef4444; border-radius:50%; animation:blink 1.2s ease-in-out infinite; }
        @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.2;} }

        .body-layout { flex:1; display:flex; overflow:hidden; }

        /* Canvas */
        .canvas-wrap {
            flex:1; position:relative; background:#000;
            display:flex; align-items:center; justify-content:center; overflow:hidden;
        }
        #liveCanvas { max-width:100%; max-height:100%; display:block; object-fit:contain; }
        #video, #capCanvas { display:none; }
        .empty-state {
            position:absolute; inset:0;
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            gap:12px; color:#2a2a50; pointer-events:none;
        }
        .empty-state .e-icon { font-size:64px; }
        .empty-state p { font-size:15px; }
        .fps-chip {
            position:absolute; top:14px; right:14px;
            background:rgba(0,0,0,.75); border:1px solid #2a2a50;
            border-radius:6px; padding:3px 10px;
            font-size:12px; font-weight:700; color:#818cf8; display:none;
        }

        /* Sidebar */
        .sidebar {
            width:340px; flex-shrink:0; background:#11112a;
            border-left:1px solid #1e1e40;
            display:flex; flex-direction:column; overflow:hidden;
        }
        .tab-bar { display:flex; border-bottom:1px solid #1e1e40; flex-shrink:0; }
        .tab-btn {
            flex:1; padding:13px 8px; background:none; border:none;
            color:#44446a; font-family:'Inter',sans-serif;
            font-size:13px; font-weight:600; cursor:pointer;
            border-bottom:2px solid transparent; transition:all .2s;
        }
        .tab-btn.active { color:#818cf8; border-bottom-color:#818cf8; }
        .panel { flex:1; overflow-y:auto; display:none; flex-direction:column; }
        .panel.active { display:flex; }
        .sec { padding:18px 18px 14px; border-bottom:1px solid #1a1a38; }
        .sec-title {
            font-size:10px; font-weight:700; text-transform:uppercase;
            letter-spacing:1.5px; color:#3a3a60; margin-bottom:12px;
        }

        /* Traffic Light */
        .tl-widget {
            background:#16163c; border:1px solid #20204a;
            border-radius:14px; padding:16px;
            display:flex; align-items:flex-start; gap:16px;
            transition:border-color 0.4s ease;
        }
        .tl-widget.state-red    { border-color:#ef444488; }
        .tl-widget.state-yellow { border-color:#f59e0b88; }
        .tl-widget.state-green  { border-color:#22c55e88; }
        .tl-pole {
            display:flex; flex-direction:column; align-items:center; gap:8px;
            background:#0d0d22; border:1px solid #20204a;
            border-radius:10px; padding:10px 8px; flex-shrink:0;
        }
        .tl-bulb {
            width:28px; height:28px; border-radius:50%;
            background:#1a1a35; border:2px solid #22224a; transition:all .3s ease;
        }
        .tl-bulb.lit-red    { background:#ef4444; border-color:#fca5a5; box-shadow:0 0 16px #ef444499,0 0 32px #ef444444; }
        .tl-bulb.lit-yellow { background:#f59e0b; border-color:#fde68a; box-shadow:0 0 16px #f59e0b99,0 0 32px #f59e0b44; }
        .tl-bulb.lit-green  { background:#22c55e; border-color:#86efac; box-shadow:0 0 16px #22c55e99,0 0 32px #22c55e44; }
        .tl-info { flex:1; min-width:0; }
        .tl-state-name { font-size:20px; font-weight:800; color:#e0e0f0; margin-bottom:2px; }

        /* Action Alerts */
        .action-alert {
            border-radius:10px; padding:11px 13px; margin-top:10px;
            display:none; animation:slideDown .35s ease;
        }
        @keyframes slideDown { from{opacity:0;transform:translateY(-6px);} to{opacity:1;transform:translateY(0);} }
        .action-alert.visible { display:block; }
        .alert-title {
            font-size:13px; font-weight:800; letter-spacing:.5px;
            margin-bottom:5px; display:flex; align-items:center; gap:7px;
        }
        .alert-desc { font-size:11.5px; line-height:1.65; font-weight:500; }

        .alert-red    { background:#1e0a0a; border:1px solid #ef444466; }
        .alert-red    .alert-title { color:#ef4444; }
        .alert-red    .alert-desc  { color:#b06060; }
        .alert-yellow { background:#1e160a; border:1px solid #f59e0b66; }
        .alert-yellow .alert-title { color:#f59e0b; }
        .alert-yellow .alert-desc  { color:#a07840; }
        .alert-green  { background:#0a1e10; border:1px solid #22c55e66; }
        .alert-green  .alert-title { color:#22c55e; }
        .alert-green  .alert-desc  { color:#408060; }
        .alert-none   { background:#14142a; border:1px solid #33335a; }
        .alert-none   .alert-title { color:#818cf8; }
        .alert-none   .alert-desc  { color:#44446a; }

        /* Signs */
        .signs-container { display:flex; flex-direction:column; gap:7px; }
        .sign-chip {
            display:flex; align-items:center; gap:9px;
            background:#16163c; border:1px solid #20204a;
            border-radius:8px; padding:9px 12px;
            font-size:13px; font-weight:600; animation:fadeSlide .25s ease;
        }
        @keyframes fadeSlide { from{opacity:0;transform:translateX(8px);} to{opacity:1;transform:translateX(0);} }
        .sign-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
        .no-sign { font-size:13px; color:#2e2e55; padding:6px 0; }

        /* Stats */
        .stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
        .stat-box { background:#16163c; border:1px solid #20204a; border-radius:10px; padding:11px 13px; }
        .stat-lbl { font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:#33335a; }
        .stat-val { font-size:20px; font-weight:800; color:#818cf8; margin-top:3px; line-height:1; }
        .stat-unit { font-size:10px; color:#444466; }
        .slider-row { display:flex; align-items:center; gap:8px; margin-top:12px; }
        .slider-row label { font-size:11px; color:#44446a; white-space:nowrap; }
        input[type=range] { flex:1; accent-color:#818cf8; cursor:pointer; }
        .slider-val { font-size:11px; font-weight:700; color:#818cf8; min-width:38px; text-align:right; }

        /* Buttons */
        .btn-stack { display:flex; flex-direction:column; gap:8px; }
        .btn {
            width:100%; padding:12px; border:none; border-radius:10px;
            font-family:'Inter',sans-serif; font-size:14px; font-weight:700;
            cursor:pointer; transition:all .2s;
            display:flex; align-items:center; justify-content:center; gap:8px;
        }
        .btn-primary { background:linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; }
        .btn-primary:hover:not(:disabled) { transform:translateY(-1px); box-shadow:0 8px 20px #6366f144; }
        .btn-warn   { background:#1a1a38; border:1px solid #f59e0b55; color:#f59e0b; }
        .btn-warn:hover:not(:disabled)   { background:#f59e0b18; }
        .btn-danger { background:#1a1a38; border:1px solid #ef444455; color:#ef4444; }
        .btn-danger:hover:not(:disabled) { background:#ef444418; }
        .btn:disabled { opacity:.3; cursor:not-allowed; }

        /* Upload */
        .upload-zone {
            border:2px dashed #20204a; border-radius:10px;
            padding:28px 16px; text-align:center; cursor:pointer;
            transition:all .2s; background:#16163c;
        }
        .upload-zone:hover { border-color:#818cf8; background:#18184a; }
        .upload-zone.over  { border-color:#c084fc; background:#1e1650; }
        .upload-zone .u-icon { font-size:30px; margin-bottom:8px; }
        .upload-zone .u-txt  { font-size:12px; color:#44446a; line-height:1.6; }
        input[type=file] { display:none; }
        .err-box {
            background:#1a0a14; border:1px solid #ef444433;
            color:#ef4444; border-radius:8px; padding:9px 12px;
            font-size:12px; margin-top:10px; display:none;
        }
        .u-result { display:none; }

        ::-webkit-scrollbar { width:3px; }
        ::-webkit-scrollbar-track { background:#0d0d1a; }
        ::-webkit-scrollbar-thumb { background:#20204a; border-radius:4px; }
    </style>
</head>
<body>

<header>
    <span class="logo">🚦</span>
    <span class="app-title">Traffic Detection System</span>
    <div class="live-pill" id="livePill"><div class="live-dot"></div> LIVE</div>
</header>

<div class="body-layout">

    <!-- Canvas -->
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

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="tab-bar">
            <button class="tab-btn active" id="tabLive"   onclick="switchTab('live')">📷 Live</button>
            <button class="tab-btn"        id="tabUpload" onclick="switchTab('upload')">📸 Upload</button>
        </div>

        <!-- LIVE PANEL -->
        <div class="panel active" id="panelLive">

            <div class="sec">
                <div class="sec-title">Traffic Light</div>
                <div class="tl-widget" id="tlWidget">
                    <div class="tl-pole">
                        <div class="tl-bulb" id="bRed"></div>
                        <div class="tl-bulb" id="bYellow"></div>
                        <div class="tl-bulb" id="bGreen"></div>
                    </div>
                    <div class="tl-info">
                        <div class="tl-state-name" id="tlStateName">—</div>

                        <div class="action-alert alert-red" id="alertRed">
                            <div class="alert-title">🛑 &nbsp;STOP!</div>
                            <div class="alert-desc">
                                Red light detected. You must stop your vehicle
                                completely and wait behind the stop line until
                                the signal turns green.
                            </div>
                        </div>

                        <div class="action-alert alert-yellow" id="alertYellow">
                            <div class="alert-title">⚠️ &nbsp;PREPARE TO STOP</div>
                            <div class="alert-desc">
                                Amber light — the signal is about to turn red.
                                Slow down and stop if it is safe to do so.
                                Consider road conditions, braking distance,
                                and vehicles behind you before deciding.
                            </div>
                        </div>

                        <div class="action-alert alert-green" id="alertGreen">
                            <div class="alert-title">✅ &nbsp;GO — SAFE TO PROCEED</div>
                            <div class="alert-desc">
                                Green light detected. You may proceed safely.
                                Always check for pedestrians and cross-traffic
                                before moving through the intersection.
                            </div>
                        </div>

                        <div class="action-alert alert-none visible" id="alertNone">
                            <div class="alert-title">🔍 &nbsp;Scanning...</div>
                            <div class="alert-desc">
                                No traffic light detected in the current frame.
                                Point the camera at a traffic signal to begin.
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="sec">
                <div class="sec-title">Detected Signs</div>
                <div class="signs-container" id="signsCont">
                    <p class="no-sign">No signs detected</p>
                </div>
            </div>

            <div class="sec">
                <div class="sec-title">Performance</div>
                <div class="stats-grid">
                    <div class="stat-box"><div class="stat-lbl">Frames</div><div class="stat-val" id="sFrames">0</div></div>
                    <div class="stat-box"><div class="stat-lbl">Latency</div><div class="stat-val" id="sLatency">—</div><span class="stat-unit">ms</span></div>
                    <div class="stat-box"><div class="stat-lbl">FPS</div><div class="stat-val" id="sFps">—</div></div>
                    <div class="stat-box"><div class="stat-lbl">Signs</div><div class="stat-val" id="sSigns">0</div></div>
                </div>
                <div class="slider-row">
                    <label>Interval</label>
                    <input type="range" id="iSlider" min="200" max="2000" step="100" value="500" oninput="setIntervalMs(this.value)">
                    <span class="slider-val" id="iVal">500ms</span>
                </div>
            </div>

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

        <!-- UPLOAD PANEL -->
        <div class="panel" id="panelUpload">
            <div class="sec">
                <div class="sec-title">Upload Image</div>
                <div class="upload-zone" id="uploadZone">
                    <div class="u-icon">📸</div>
                    <div class="u-txt" id="uTxt">
                        Click or drag &amp; drop an image<br>
                        <small style="color:#2e2e55">PNG · JPG · BMP &nbsp;(max 16 MB)</small>
                    </div>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
                <div class="err-box" id="uploadErr"></div>
            </div>

            <div class="sec u-result" id="uResultSec">
                <div class="sec-title">Result</div>
                <div class="tl-widget" id="uTlWidget">
                    <div class="tl-pole">
                        <div class="tl-bulb" id="uBRed"></div>
                        <div class="tl-bulb" id="uBYellow"></div>
                        <div class="tl-bulb" id="uBGreen"></div>
                    </div>
                    <div class="tl-info">
                        <div class="tl-state-name" id="uTlStateName">—</div>
                        <div class="action-alert alert-red"    id="uAlertRed">
                            <div class="alert-title">🛑 &nbsp;STOP!</div>
                            <div class="alert-desc">Red light — vehicle must stop and wait.</div>
                        </div>
                        <div class="action-alert alert-yellow" id="uAlertYellow">
                            <div class="alert-title">⚠️ &nbsp;PREPARE TO STOP</div>
                            <div class="alert-desc">Amber — slow down, stop if safe to do so.</div>
                        </div>
                        <div class="action-alert alert-green"  id="uAlertGreen">
                            <div class="alert-title">✅ &nbsp;GO — SAFE TO PROCEED</div>
                            <div class="alert-desc">Green light — proceed safely, check pedestrians.</div>
                        </div>
                        <div class="action-alert alert-none visible" id="uAlertNone">
                            <div class="alert-title">🔍 &nbsp;No Signal Detected</div>
                            <div class="alert-desc">No traffic light found in the uploaded image.</div>
                        </div>
                    </div>
                </div>
                <div class="signs-container" id="uSignsCont" style="margin-top:12px;"></div>
            </div>
        </div>

    </div>
</div>

<script>
/* ── State ── */
let camStream=null,detectTimer=null,fpsTimer2=null;
let isLive=false,paused=false,frames=0,fpsBucket=0,fps=0;
let intervalMs=500,requesting=false,lastSig='';

const video     =document.getElementById('video');
const capCanvas =document.getElementById('capCanvas');
const liveCanvas=document.getElementById('liveCanvas');
const liveCtx   =liveCanvas.getContext('2d');
const capCtx    =capCanvas.getContext('2d');

/* ── Tabs ── */
function switchTab(tab){
    ['live','upload'].forEach(t=>{
        document.getElementById('panel'+t[0].toUpperCase()+t.slice(1)).classList.toggle('active',t===tab);
        document.getElementById('tab'  +t[0].toUpperCase()+t.slice(1)).classList.toggle('active',t===tab);
    });
}

/* ── Live Detection ── */
async function startLive(){
    clearErr('live');
    try{
        camStream=await navigator.mediaDevices.getUserMedia(
            {video:{width:{ideal:640},height:{ideal:480},facingMode:'user'}});
        video.srcObject=camStream;
        await new Promise(r=>{video.onloadedmetadata=r;});
        video.play();
        liveCanvas.width=capCanvas.width=video.videoWidth||640;
        liveCanvas.height=capCanvas.height=video.videoHeight||480;
        document.getElementById('emptyState').style.display='none';
        document.getElementById('livePill').classList.add('visible');
        document.getElementById('fpsChip').style.display='block';
        setButtons(false,true,true);
        isLive=true;paused=false;lastSig='';
        renderLoop();
        detectTimer=setInterval(sendFrame,intervalMs);
        fpsTimer2=setInterval(()=>{
            fps=fpsBucket;fpsBucket=0;
            document.getElementById('fpsChip').textContent=fps+' FPS';
            document.getElementById('sFps').textContent=fps;
        },1000);
    }catch(e){showErr('live','Camera access denied — please allow permission.');}
}

function renderLoop(){
    if(!isLive)return;
    if(video.readyState>=2){liveCtx.drawImage(video,0,0,liveCanvas.width,liveCanvas.height);fpsBucket++;}
    requestAnimationFrame(renderLoop);
}

function sendFrame(){
    if(!isLive||paused||requesting)return;
    if(video.readyState<2)return;
    capCtx.drawImage(video,0,0,capCanvas.width,capCanvas.height);
    const t0=performance.now();requesting=true;
    capCanvas.toBlob(blob=>{
        if(!blob){requesting=false;return;}
        const fd=new FormData();fd.append('file',blob,'frame.jpg');
        fetch('/api/detect',{method:'POST',body:fd})
        .then(r=>r.json()).then(data=>{
            requesting=false;
            frames++;
            document.getElementById('sFrames').textContent=frames;
            document.getElementById('sLatency').textContent=Math.round(performance.now()-t0);
            if(data.success){
                updateLight(data.traffic_light,'b','tlStateName','tlWidget',
                    'alertRed','alertYellow','alertGreen','alertNone',true);
                updateSigns(data.traffic_signs,'signsCont','sSigns');
                if(data.image){const img=new Image();img.onload=()=>liveCtx.drawImage(img,0,0,liveCanvas.width,liveCanvas.height);img.src=data.image;}
            }
        }).catch(()=>{requesting=false;});
    },'image/jpeg',0.75);
}

function togglePause(){paused=!paused;document.getElementById('btnPause').textContent=paused?'▶  Resume':'⏸  Pause';}

function stopLive(){
    isLive=false;requesting=false;
    clearInterval(detectTimer);clearInterval(fpsTimer2);
    if(camStream)camStream.getTracks().forEach(t=>t.stop());
    camStream=null;
    liveCtx.clearRect(0,0,liveCanvas.width,liveCanvas.height);
    document.getElementById('emptyState').style.display='flex';
    document.getElementById('livePill').classList.remove('visible');
    document.getElementById('fpsChip').style.display='none';
    setButtons(true,false,false);
    frames=0;
    ['sFrames','sLatency','sFps','sSigns'].forEach((id,i)=>document.getElementById(id).textContent=i===0?0:'—');
    resetLight('b','tlStateName','tlWidget','alertRed','alertYellow','alertGreen','alertNone');
    document.getElementById('signsCont').innerHTML='<p class="no-sign">No signs detected</p>';
}

function setIntervalMs(v){
    intervalMs=parseInt(v);document.getElementById('iVal').textContent=v+'ms';
    if(isLive){clearInterval(detectTimer);detectTimer=setInterval(sendFrame,intervalMs);}
}

/* ── Traffic Light & Alerts ── */
const LABELS={red:'RED LIGHT',yellow:'YELLOW LIGHT',green:'GREEN LIGHT',none:'NO SIGNAL',unknown:'NO SIGNAL'};

function updateLight(light,bPfx,nameId,widgetId,idR,idY,idG,idN,checkCache){
    const sig=(light&&light.detected)?light.detected.toLowerCase():'none';
    if(checkCache&&sig===lastSig)return;
    if(checkCache)lastSig=sig;

    ['Red','Yellow','Green'].forEach(c=>{
        const el=document.getElementById(bPfx+c);
        if(el)el.className='tl-bulb'+(sig===c.toLowerCase()?' lit-'+c.toLowerCase():'');
    });
    const nEl=document.getElementById(nameId);if(nEl)nEl.textContent=LABELS[sig]||'NO SIGNAL';
    const wEl=document.getElementById(widgetId);
    if(wEl)wEl.className='tl-widget'+(sig!=='none'&&sig!=='unknown'?' state-'+sig:'');

    const map={red:idR,yellow:idY,green:idG,none:idN,unknown:idN};
    [idR,idY,idG,idN].forEach(id=>{const el=document.getElementById(id);if(el)el.classList.remove('visible');});
    const target=document.getElementById(map[sig]||idN);if(target)target.classList.add('visible');
}

function resetLight(bPfx,nameId,widgetId,idR,idY,idG,idN){
    ['Red','Yellow','Green'].forEach(c=>{const el=document.getElementById(bPfx+c);if(el)el.className='tl-bulb';});
    const nEl=document.getElementById(nameId);if(nEl)nEl.textContent='—';
    const wEl=document.getElementById(widgetId);if(wEl)wEl.className='tl-widget';
    [idR,idY,idG].forEach(id=>{const el=document.getElementById(id);if(el)el.classList.remove('visible');});
    const noneEl=document.getElementById(idN);if(noneEl)noneEl.classList.add('visible');
    lastSig='';
}

/* ── Signs ── */
const SC={'stop':'#ef4444','yield':'#f59e0b','speed':'#3b82f6','park':'#8b5cf6','no entry':'#ef4444'};
function sCol(n){n=n.toLowerCase();for(const[k,v]of Object.entries(SC))if(n.includes(k))return v;return'#22c55e';}
function updateSigns(sd,cId,countId){
    const cont=document.getElementById(cId);
    const signs=(sd&&sd.signs)?sd.signs:[];
    if(countId)document.getElementById(countId).textContent=signs.length;
    if(!signs.length){cont.innerHTML='<p class="no-sign">No signs detected</p>';return;}
    const key=signs.join(',');if(cont.dataset.key===key)return;cont.dataset.key=key;
    cont.innerHTML=signs.map(s=>`<div class="sign-chip"><div class="sign-dot" style="background:${sCol(s)}"></div><span>${s}</span></div>`).join('');
}

/* ── Upload ── */
const uploadZone=document.getElementById('uploadZone');
const fileInput=document.getElementById('fileInput');
uploadZone.addEventListener('click',()=>fileInput.click());
uploadZone.addEventListener('dragover',e=>{e.preventDefault();uploadZone.classList.add('over');});
uploadZone.addEventListener('dragleave',()=>uploadZone.classList.remove('over'));
uploadZone.addEventListener('drop',e=>{e.preventDefault();uploadZone.classList.remove('over');if(e.dataTransfer.files.length){fileInput.files=e.dataTransfer.files;doUpload();}});
fileInput.addEventListener('change',doUpload);

function doUpload(){
    const file=fileInput.files[0];
    if(!file||!file.type.startsWith('image/')){showErr('upload','Please select a valid image.');return;}
    clearErr('upload');
    const uTxt=document.getElementById('uTxt');
    uTxt.textContent='⏳ Processing...';
    const fd=new FormData();fd.append('file',file);
    fetch('/api/detect',{method:'POST',body:fd}).then(r=>r.json()).then(data=>{
        uTxt.innerHTML='Click or drag &amp; drop an image<br><small style="color:#2e2e55">PNG · JPG · BMP  (max 16 MB)</small>';
        if(!data.success){showErr('upload',data.error||'Detection failed');return;}
        if(data.image){
            const img=new Image();
            img.onload=()=>{document.getElementById('emptyState').style.display='none';liveCanvas.width=img.naturalWidth;liveCanvas.height=img.naturalHeight;liveCtx.drawImage(img,0,0);};
            img.src=data.image;
        }
        document.getElementById('uResultSec').style.display='';
        updateLight(data.traffic_light,'uB','uTlStateName','uTlWidget','uAlertRed','uAlertYellow','uAlertGreen','uAlertNone',false);
        updateSigns(data.traffic_signs,'uSignsCont',null);
    }).catch(e=>{uTxt.innerHTML='Click or drag &amp; drop an image';showErr('upload','Request failed: '+e.message);});
}

/* ── Misc ── */
function setButtons(s,p,st){document.getElementById('btnStart').disabled=!s;document.getElementById('btnPause').disabled=!p;document.getElementById('btnStop').disabled=!st;}
function showErr(ctx,msg){const el=document.getElementById(ctx==='live'?'liveErr':'uploadErr');el.textContent='❌ '+msg;el.style.display='block';}
function clearErr(ctx){document.getElementById(ctx==='live'?'liveErr':'uploadErr').style.display='none';}
</script>
</body>
</html>"""
    return html, 200


if __name__ == '__main__':
    app.run(debug=True)
