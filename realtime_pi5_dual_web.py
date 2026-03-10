"""
Raspberry Pi 5 Dual CSI Camera Real-time Crack Detection with Flask Web Streaming
- Streams annotated video to web browser
- Access at http://raspberrypi-ip:5000
"""

import os
import csv
import time
import json
import threading
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from flask import Flask, Response, render_template_string

# ---------------- ENV ----------------
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

API_KEY = os.getenv("RF_API_KEY", "").strip()
WORKSPACE = os.getenv("RF_WORKSPACE", "").strip()
WORKFLOW_ID = os.getenv("RF_WORKFLOW_ID", "").strip()

if not API_KEY or not WORKSPACE or not WORKFLOW_ID:
    raise ValueError("Missing .env vars: RF_API_KEY, RF_WORKSPACE, RF_WORKFLOW_ID")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)

# ---------------- SETTINGS ----------------
CONF_THRESH = float(os.getenv("RF_CONF", "0.5"))
INFER_FPS = 1.0
SAVE_COOLDOWN_S = 0.5
ONLY_CLASS = ""

ENABLE_PREPROCESSING = True
ENABLE_PERSISTENCE = False
PERSISTENCE_FRAMES = 3
BLUR_THRESHOLD = 5.0  # Lowered for testing - camera getting ~7.0
MIN_CRACK_AREA = 100

SEVERITY_CRITICAL = 0.85
SEVERITY_HIGH = 0.70
SEVERITY_MEDIUM = 0.55

BOOLEAN_DURATION_S = 1.0

# ---------- PIPELINE LOCALIZATION SETTINGS ----------
# IMPORTANT: These settings enable crack position estimation for live inspection
# Position is estimated based on elapsed time assuming constant robot speed

# Set your pipeline length in meters
PIPELINE_LENGTH_METERS = 100.0  # Change to your actual pipeline length

# Set estimated inspection duration in seconds (time to traverse full pipeline)
# Example: If robot takes 10 minutes to inspect 100m pipe, set to 600.0
ESTIMATED_INSPECTION_DURATION_SEC = 600.0  # Change based on your robot speed

# Alternative: If you know robot speed, calculate duration:
# ROBOT_SPEED_MPS = 0.167  # meters per second (e.g., 10 m/min = 0.167 m/s)
# ESTIMATED_INSPECTION_DURATION_SEC = PIPELINE_LENGTH_METERS / ROBOT_SPEED_MPS

# Enable/disable position tracking
ENABLE_POSITION_TRACKING = True

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_FPS = 30

CAMERA_0_ID = 0
CAMERA_1_ID = 1

DASHBOARD_INTERVAL_S = 5.0

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

OUT_BASE = Path("data/realtime_results")
FOUND_DIR_CAM0 = OUT_BASE / "camera0_found"
FOUND_DIR_CAM1 = OUT_BASE / "camera1_found"
REALTIME_FOUND_DIR_CAM0 = OUT_BASE / "camera0_realtime"
REALTIME_FOUND_DIR_CAM1 = OUT_BASE / "camera1_realtime"
REPORTS_DIR = OUT_BASE / "reports"

for p in [FOUND_DIR_CAM0, FOUND_DIR_CAM1, REALTIME_FOUND_DIR_CAM0, REALTIME_FOUND_DIR_CAM1, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------- HELPERS ----------------
def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def estimate_crack_position(elapsed_sec: float, estimated_duration_sec: float, pipeline_length_m: float) -> float:
    """
    Estimate crack position along the pipeline based on elapsed time.
    
    IMPORTANT: This is an APPROXIMATE estimate assuming constant robot speed.
    The actual position may vary if the robot speed changes during inspection.
    
    Formula: position_m = pipeline_length_m * (elapsed_time / total_inspection_time)
    
    Args:
        elapsed_sec: Time elapsed since inspection start (seconds)
        estimated_duration_sec: Expected total inspection duration (seconds)
        pipeline_length_m: Total pipeline length in meters
    
    Returns:
        Estimated position in meters from pipe entrance
    """
    if not ENABLE_POSITION_TRACKING or estimated_duration_sec <= 0:
        return 0.0
    
    # Clamp elapsed time to valid range
    elapsed_sec = max(0.0, min(elapsed_sec, estimated_duration_sec))
    
    # Proportional calculation based on time progress
    # Estimated crack location from pipe entrance
    # position_m = pipeline_length_m * (elapsed_time / total_inspection_time)
    position_m = pipeline_length_m * (elapsed_sec / estimated_duration_sec)
    
    return position_m


def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, list):
        for item in result:
            preds = extract_predictions(item)
            if preds:
                return preds
        return []
    if isinstance(result, dict):
        preds = result.get("predictions")
        if isinstance(preds, list):
            return preds
        for v in result.values():
            preds = extract_predictions(v)
            if preds:
                return preds
    return []


def pred_conf(p: Dict[str, Any]) -> float:
    return float(p.get("confidence", p.get("score", 0.0)) or 0.0)


def pred_class(p: Dict[str, Any]) -> str:
    return str(p.get("class", p.get("class_name", p.get("label", ""))) or "")


def filter_preds(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in preds:
        if pred_conf(p) < CONF_THRESH:
            continue
        if ONLY_CLASS and pred_class(p).lower() != ONLY_CLASS.lower():
            continue
        out.append(p)
    return out


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if not ENABLE_PREPROCESSING:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
    enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(enhanced_bgr, 0.7, frame, 0.3, 0)
    return blended


def check_frame_quality(frame: np.ndarray) -> Tuple[bool, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (lap_var >= BLUR_THRESHOLD), float(lap_var)


def calculate_crack_area(pred: Dict[str, Any]) -> float:
    w = float(pred.get("width", 0))
    h = float(pred.get("height", 0))
    return w * h


def classify_severity(confidence: float) -> str:
    if confidence >= SEVERITY_CRITICAL:
        return "CRITICAL"
    if confidence >= SEVERITY_HIGH:
        return "HIGH"
    if confidence >= SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def draw_detections(frame: np.ndarray, preds: List[Dict[str, Any]], position_m: float = 0.0) -> np.ndarray:
    """Draw detections with optional position information"""
    for pred in preds:
        x = int(pred.get("x", 0))
        y = int(pred.get("y", 0))
        w = int(pred.get("width", 0))
        h = int(pred.get("height", 0))

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        conf = pred_conf(pred)
        severity = classify_severity(conf)

        if severity == "CRITICAL":
            color, thickness = (0, 0, 255), 3
        elif severity == "HIGH":
            color, thickness = (0, 165, 255), 3
        elif severity == "MEDIUM":
            color, thickness = (0, 255, 255), 2
        else:
            color, thickness = (0, 255, 0), 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        area = calculate_crack_area(pred)
        if ENABLE_POSITION_TRACKING and position_m > 0:
            label = f"Crack at {position_m:.2f}m [{severity}] {conf:.2f}"
        else:
            label = f"{pred_class(pred)} [{severity}] {conf:.2f} ({int(area)}px)"
        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th_text - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


# ---------------- STATE ----------------
class CameraState:
    def __init__(self, camera_id: int, found_dir: Path, realtime_dir: Path):
        self.camera_id = camera_id
        self.found_dir = found_dir
        self.realtime_dir = realtime_dir

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.latest_annotated_frame = None
        self.annotated_lock = threading.Lock()

        self.latest_result = {"status": "idle", "best": 0.0, "count": 0, "predictions": [], "blur_score": 0.0}
        self.result_lock = threading.Lock()

        self.detection_history = deque(maxlen=PERSISTENCE_FRAMES)
        self.history_lock = threading.Lock()

        self.boolean_on = False
        self.boolean_until = 0.0
        self.boolean_lock = threading.Lock()

        # Position tracking
        self.start_time = time.time()
        self.crack_counter = 0
        self.position_lock = threading.Lock()
        
        # CSV report file
        if ENABLE_POSITION_TRACKING:
            csv_filename = REPORTS_DIR / f"cam{camera_id}_crack_report_{stamp()}.csv"
            self.csv_file = open(csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "crack_id", "elapsed_sec", "position_m", "confidence", 
                "severity", "area_px", "class", "image_name", "timestamp"
            ])
            self.csv_lock = threading.Lock()
        else:
            self.csv_file = None
            self.csv_writer = None

        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "skipped_blurry": 0,
            "detections_found": 0,
            "critical_cracks": 0,
            "high_cracks": 0,
            "medium_cracks": 0,
            "low_cracks": 0,
            "total_saved": 0,
            "inference_errors": 0,
        }
        self.stats_lock = threading.Lock()

        self.stop_flag = False

    def update_detection_history(self, has_detection: bool):
        with self.history_lock:
            self.detection_history.append(has_detection)

    def check_detection_persistence(self) -> bool:
        if not ENABLE_PERSISTENCE:
            return True
        with self.history_lock:
            if len(self.detection_history) < PERSISTENCE_FRAMES:
                return False
            return all(self.detection_history)

    def set_boolean(self):
        with self.boolean_lock:
            self.boolean_on = True
            self.boolean_until = time.time() + BOOLEAN_DURATION_S

    def get_boolean(self) -> bool:
        with self.boolean_lock:
            if self.boolean_on and time.time() > self.boolean_until:
                self.boolean_on = False
            return self.boolean_on
    
    def get_elapsed_time(self) -> float:
        """Get time elapsed since inspection start"""
        return time.time() - self.start_time
    
    def get_estimated_position(self) -> float:
        """Get current estimated position in pipeline"""
        if not ENABLE_POSITION_TRACKING:
            return 0.0
        elapsed = self.get_elapsed_time()
        return estimate_crack_position(elapsed, ESTIMATED_INSPECTION_DURATION_SEC, PIPELINE_LENGTH_METERS)
    
    def increment_crack_counter(self) -> int:
        """Increment and return crack counter"""
        with self.position_lock:
            self.crack_counter += 1
            return self.crack_counter
    
    def write_crack_to_csv(self, crack_id: int, elapsed_sec: float, position_m: float, 
                           conf: float, severity: str, area: float, class_name: str, 
                           image_name: str, timestamp: float):
        """Write crack detection to CSV report"""
        if self.csv_writer is not None:
            with self.csv_lock:
                self.csv_writer.writerow([
                    crack_id, f"{elapsed_sec:.2f}", f"{position_m:.2f}", 
                    f"{conf:.3f}", severity, int(area), class_name, 
                    image_name, timestamp
                ])
                self.csv_file.flush()
    
    def close_csv(self):
        """Close CSV file"""
        if self.csv_file is not None:
            self.csv_file.close()


# ---------------- CAPTURE (rpicam pipe) ----------------
def _spawn_rpicam_mjpeg(camera_id: int) -> subprocess.Popen:
    cmd = [
        "rpicam-vid",
        "--camera", str(camera_id),
        "-n",
        "--codec", "mjpeg",
        "--width", str(CAMERA_WIDTH),
        "--height", str(CAMERA_HEIGHT),
        "--framerate", str(CAPTURE_FPS),
        "-t", "0",
        "-o", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


def camera_capture_thread(cam_state: CameraState, rpicam_camera_id: int):
    proc = _spawn_rpicam_mjpeg(rpicam_camera_id)

    data = bytearray()
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    print(f"[CAM{cam_state.camera_id}] rpicam-vid started (camera={rpicam_camera_id})")

    try:
        while not cam_state.stop_flag:
            chunk = proc.stdout.read(4096)
            if not chunk:
                time.sleep(0.01)
                continue

            data.extend(chunk)

            start = data.find(SOI)
            if start == -1:
                if len(data) > 2_000_000:
                    data.clear()
                continue

            end = data.find(EOI, start + 2)
            if end == -1:
                continue

            jpg = data[start:end + 2]
            del data[:end + 2]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            with cam_state.frame_lock:
                cam_state.latest_frame = frame

    finally:
        cam_state.stop_flag = True
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        print(f"[CAM{cam_state.camera_id}] capture stopped")


# ---------------- INFERENCE ----------------
def inference_loop(cam_state: CameraState):
    min_interval = 1.0 / max(INFER_FPS, 0.1)
    last_infer_time = 0.0
    last_saved_found = 0.0

    while not cam_state.stop_flag:
        now = time.time()
        if now - last_infer_time < min_interval:
            time.sleep(0.01)
            continue

        with cam_state.frame_lock:
            frame = None if cam_state.latest_frame is None else cam_state.latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        with cam_state.stats_lock:
            cam_state.stats["total_frames"] += 1

        is_good, blur_score = check_frame_quality(frame)
        if not is_good:
            with cam_state.stats_lock:
                cam_state.stats["skipped_blurry"] += 1
            cam_state.update_detection_history(False)
            with cam_state.result_lock:
                cam_state.latest_result["status"] = "blurry"
                cam_state.latest_result["blur_score"] = blur_score
            
            elapsed = cam_state.get_elapsed_time()
            position_m = cam_state.get_estimated_position()
            
            display_frame = frame.copy()
            status_text = f"CAM{cam_state.camera_id} | BLURRY (blur={blur_score:.1f})"
            if ENABLE_POSITION_TRACKING:
                status_text += f" | Pos: {position_m:.2f}m"
            cv2.putText(display_frame, status_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            with cam_state.annotated_lock:
                cam_state.latest_annotated_frame = display_frame
            
            print(f"[CAM{cam_state.camera_id}] skipped blurry frame (blur={blur_score:.1f} < {BLUR_THRESHOLD})")
            time.sleep(0.02)
            continue

        last_infer_time = now
        processed = preprocess_frame(frame)

        ok, jpg = cv2.imencode(".jpg", processed)
        if not ok:
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
                f.write(jpg.tobytes())
                f.flush()

                result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW_ID,
                    images={"image": f.name},
                    use_cache=True,
                )

            with cam_state.stats_lock:
                cam_state.stats["processed_frames"] += 1

            preds = filter_preds(extract_predictions(result))
            preds = [p for p in preds if calculate_crack_area(p) >= MIN_CRACK_AREA]

            best = max((pred_conf(p) for p in preds), default=0.0)
            count = len(preds)
            found = count > 0

            cam_state.update_detection_history(found)

            if found:
                cam_state.set_boolean()

            bool_state = cam_state.get_boolean()
            elapsed = cam_state.get_elapsed_time()
            position_m = cam_state.get_estimated_position()
            classes_seen = [pred_class(p) for p in preds] if preds else []
            
            display_frame = draw_detections(frame.copy(), preds, position_m)
            status_text = f"CAM{cam_state.camera_id} | {'CRACK!' if found else 'OK'} | dets={count} conf={best:.2f}"
            if ENABLE_POSITION_TRACKING:
                status_text += f" | {position_m:.2f}m / {PIPELINE_LENGTH_METERS:.0f}m"
            status_color = (0, 0, 255) if found else (0, 255, 0)
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            with cam_state.annotated_lock:
                cam_state.latest_annotated_frame = display_frame
            
            print(
                f"[CAM{cam_state.camera_id}] found={found} count={count} best={best:.2f} "
                f"blur={blur_score:.1f} pos={position_m:.2f}m boolean={bool_state} classes={classes_seen}"
            )

            if found:
                with cam_state.stats_lock:
                    cam_state.stats["detections_found"] += 1
                    for p in preds:
                        sev = classify_severity(pred_conf(p))
                        if sev == "CRITICAL":
                            cam_state.stats["critical_cracks"] += 1
                        elif sev == "HIGH":
                            cam_state.stats["high_cracks"] += 1
                        elif sev == "MEDIUM":
                            cam_state.stats["medium_cracks"] += 1
                        else:
                            cam_state.stats["low_cracks"] += 1

            with cam_state.result_lock:
                cam_state.latest_result["status"] = "found" if found else "not_found"
                cam_state.latest_result["best"] = best
                cam_state.latest_result["count"] = count
                cam_state.latest_result["predictions"] = preds
                cam_state.latest_result["blur_score"] = blur_score

            is_persistent = cam_state.check_detection_persistence()
            t = time.time()

            if found and is_persistent and (t - last_saved_found) >= SAVE_COOLDOWN_S:
                last_saved_found = t
                elapsed = cam_state.get_elapsed_time()
                position_m = cam_state.get_estimated_position()
                
                # Process each crack separately
                for pred in preds:
                    crack_id = cam_state.increment_crack_counter()
                    conf = pred_conf(pred)
                    sev = classify_severity(conf)
                    area = calculate_crack_area(pred)
                    class_name = pred_class(pred)
                    
                    # Generate filename with position
                    if ENABLE_POSITION_TRACKING:
                        name = f"cam{cam_state.camera_id}_crack{crack_id:04d}_pos{position_m:.2f}m_{stamp()}_{int(t*1000)}"
                    else:
                        name = f"cam{cam_state.camera_id}_{stamp()}_{int(t*1000)}"
                    
                    with cam_state.stats_lock:
                        cam_state.stats["total_saved"] += 1

                    annotated_raw = draw_detections(frame.copy(), [pred], position_m)
                    annotated_enhanced = draw_detections(processed.copy(), [pred], position_m)

                    metadata = {
                        "crack_id": crack_id,
                        "camera_id": cam_state.camera_id,
                        "elapsed_sec": elapsed,
                        "position_m": position_m,
                        "pipeline_length_m": PIPELINE_LENGTH_METERS,
                        "result": result,
                        "blur_score": blur_score,
                        "detections": 1,
                        "confidence": conf,
                        "severity": sev,
                        "area_px": area,
                        "class": class_name,
                        "timestamp": t,
                        "preprocessing_enabled": ENABLE_PREPROCESSING,
                        "persistence_enabled": ENABLE_PERSISTENCE,
                        "persistence_frames": PERSISTENCE_FRAMES,
                        "position_tracking_enabled": ENABLE_POSITION_TRACKING,
                    }

                    base_found = cam_state.found_dir / name
                    cv2.imwrite(str(base_found) + ".jpg", frame)
                    cv2.imwrite(str(base_found) + "_enhanced.jpg", processed)
                    cv2.imwrite(str(base_found) + "_annotated.jpg", annotated_raw)
                    cv2.imwrite(str(base_found) + "_enhanced_annotated.jpg", annotated_enhanced)
                    Path(str(base_found) + ".json").write_text(json.dumps(metadata, indent=2, default=str))

                    base_rt = cam_state.realtime_dir / name
                    cv2.imwrite(str(base_rt) + ".jpg", frame)
                    cv2.imwrite(str(base_rt) + "_enhanced.jpg", processed)
                    cv2.imwrite(str(base_rt) + "_annotated.jpg", annotated_raw)
                    cv2.imwrite(str(base_rt) + "_enhanced_annotated.jpg", annotated_enhanced)
                    Path(str(base_rt) + ".json").write_text(json.dumps(metadata, indent=2, default=str))
                    
                    # Write to CSV
                    if ENABLE_POSITION_TRACKING:
                        cam_state.write_crack_to_csv(
                            crack_id, elapsed, position_m, conf, sev, 
                            area, class_name, name + ".jpg", t
                        )

                    print(
                        f"[CAM{cam_state.camera_id}] SAVED crack#{crack_id} at {position_m:.2f}m | "
                        f"{sev} conf={conf:.2f} class={class_name}"
                    )

        except Exception as e:
            with cam_state.stats_lock:
                cam_state.stats["inference_errors"] += 1
            print(f"[CAM{cam_state.camera_id}] inference error: {type(e).__name__}: {e}")
            with cam_state.result_lock:
                cam_state.latest_result = {
                    "status": "error",
                    "best": 0.0,
                    "count": 0,
                    "predictions": [],
                    "blur_score": blur_score,
                }
            time.sleep(0.5)


# ---------------- DASHBOARD ----------------
def dashboard_thread(cam_states: List[CameraState]):
    while not all(cs.stop_flag for cs in cam_states):
        time.sleep(DASHBOARD_INTERVAL_S)
        lines = [
            "",
            "=" * 70,
            f"  DASHBOARD  |  {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
        ]
        for cs in cam_states:
            with cs.result_lock:
                status = cs.latest_result.get("status", "?")
                best = cs.latest_result.get("best", 0.0)
                count = cs.latest_result.get("count", 0)
                blur = cs.latest_result.get("blur_score", 0.0)
            bool_state = cs.get_boolean()
            with cs.stats_lock:
                s = dict(cs.stats)
            lines.append(
                f"  CAM{cs.camera_id}: status={status:>10} | dets={count} best={best:.2f} "
                f"blur={blur:.1f} | CRACK_BOOL={bool_state}"
            )
            lines.append(
                f"         processed={s['processed_frames']} saved={s['total_saved']} "
                f"blurry={s['skipped_blurry']} errors={s['inference_errors']}"
            )
            lines.append(
                f"         severity: CRITICAL={s['critical_cracks']} HIGH={s['high_cracks']} "
                f"MEDIUM={s['medium_cracks']} LOW={s['low_cracks']}"
            )
        lines.append("=" * 70)
        lines.append("")
        print("\n".join(lines))


# ---------------- FLASK WEB SERVER ----------------
app = Flask(__name__)

cam0 = None
cam1 = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pi5 Dual Camera Crack Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #00ff00;
        }
        .container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
        }
        .camera-box {
            background: #2a2a2a;
            border: 2px solid #444;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        img {
            max-width: 640px;
            width: 100%;
            border: 2px solid #666;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>🔍 Pi5 Dual Camera Crack Detection</h1>
    <div class="container">
        <div class="camera-box">
            <h2>Camera 0</h2>
            <img src="/video_feed/0" alt="Camera 0">
        </div>
        <div class="camera-box">
            <h2>Camera 1</h2>
            <img src="/video_feed/1" alt="Camera 1">
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


def generate_frames(cam_state: CameraState):
    while True:
        with cam_state.annotated_lock:
            frame = cam_state.latest_annotated_frame
        
        if frame is None:
            time.sleep(0.033)
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    if cam_id == 0:
        return Response(generate_frames(cam0), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif cam_id == 1:
        return Response(generate_frames(cam1), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid camera ID", 404


# ---------------- MAIN ----------------
def main():
    global cam0, cam1
    
    print("=" * 60)
    print("  Pi 5 SINGLE CSI Camera Crack Detection (WEB STREAMING)")
    print("=" * 60)
    print(f"  Camera 0 rpicam id : {CAMERA_0_ID}")
    print(f"  Camera 1           : DISABLED (not connected)")
    print(f"  Flask web server   : http://0.0.0.0:{FLASK_PORT}")
    print(f"  Access from browser: http://<pi-ip>:{FLASK_PORT}")
    
    if ENABLE_POSITION_TRACKING:
        print("\n  PIPELINE LOCALIZATION ENABLED")
        print(f"  Pipeline Length:   {PIPELINE_LENGTH_METERS:.1f}m")
        print(f"  Est. Duration:     {ESTIMATED_INSPECTION_DURATION_SEC:.0f}s ({ESTIMATED_INSPECTION_DURATION_SEC/60:.1f} min)")
        print(f"  Position Estimate: Based on elapsed time (approx. constant speed)")
        print("  IMPORTANT: This is an APPROXIMATE estimate, not precise odometry!")
    
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")

    cam0 = CameraState(0, FOUND_DIR_CAM0, REALTIME_FOUND_DIR_CAM0)
    cam1 = CameraState(1, FOUND_DIR_CAM1, REALTIME_FOUND_DIR_CAM1)  # Dummy, not used

    # Only start Camera 0 threads
    t_cap0 = threading.Thread(target=camera_capture_thread, args=(cam0, CAMERA_0_ID), daemon=True)
    t_inf0 = threading.Thread(target=inference_loop, args=(cam0,), daemon=True)
    t_dash = threading.Thread(target=dashboard_thread, args=([cam0],), daemon=True)  # Only cam0 in dashboard

    t_cap0.start()
    t_inf0.start()
    t_dash.start()

    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        cam0.stop_flag = True
        time.sleep(0.5)
        
        # Close CSV files
        cam0.close_csv()

        print("\n" + "=" * 60)
        print("  FINAL STATS - PIPELINE INSPECTION")
        print("=" * 60)
        
        if ENABLE_POSITION_TRACKING:
            elapsed = cam0.get_elapsed_time()
            final_position = cam0.get_estimated_position()
            print(f"\n  Pipeline Length:   {PIPELINE_LENGTH_METERS:.1f}m")
            print(f"  Inspection Time:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"  Final Position:    {final_position:.2f}m")
            print(f"  Total Cracks:      {cam0.crack_counter}")
        
        with cam0.stats_lock:
            s = dict(cam0.stats)
        print(f"\n  Camera 0:")
        print(f"    Total Frames:      {s['total_frames']}")
        print(f"    Processed:         {s['processed_frames']}")
        print(f"    Saved:             {s['total_saved']}")
        print(f"    Errors:            {s['inference_errors']}")
        print(f"    Severity: Critical={s['critical_cracks']} High={s['high_cracks']} "\
              f"Medium={s['medium_cracks']} Low={s['low_cracks']}")
        
        if ENABLE_POSITION_TRACKING:
            print(f"\n  CSV Report: {REPORTS_DIR}/cam0_crack_report_*.csv")
        
        print("\n" + "=" * 60)
        print("Done.")


if __name__ == "__main__":
    main()
