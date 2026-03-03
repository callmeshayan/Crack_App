"""
Raspberry Pi 5 Dual Camera Real-time Crack Detection
Supports two cameras running simultaneously for crack detection.
"""
import os
import time
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple
import tempfile
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# ---------- ENV ----------
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

# ---------- SETTINGS ----------
CONF_THRESH = float(os.getenv("RF_CONF", "0.5"))
INFER_FPS = 2.0                 # target inference rate per camera
SAVE_COOLDOWN_S = 0.5           # minimum time between saved frames per category
ONLY_CLASS = "crack"            # set "" to disable class filtering
SHOW_PREVIEW = True

# Enhanced detection settings
ENABLE_PREPROCESSING = True     # Enable image preprocessing for better detection
ENABLE_PERSISTENCE = True       # Only save detections that persist across frames
PERSISTENCE_FRAMES = 3          # Number of consecutive frames needed to confirm detection
BLUR_THRESHOLD = 100.0          # Skip frames with blur variance below this (Laplacian)
MIN_CRACK_AREA = 100            # Minimum crack area in pixels to consider

# Severity classification thresholds
SEVERITY_CRITICAL = 0.85        # Confidence >= 0.85: Critical
SEVERITY_HIGH = 0.70            # Confidence >= 0.70: High
SEVERITY_MEDIUM = 0.55          # Confidence >= 0.55: Medium
# Below CONF_THRESH: Filtered out

# Camera settings for Raspberry Pi 5
CAMERA_0_INDEX = 0              # First camera (e.g., /dev/video0)
CAMERA_1_INDEX = 2              # Second camera (e.g., /dev/video2)
CAMERA_WIDTH = 640              # Resolution width
CAMERA_HEIGHT = 480             # Resolution height

OUT_BASE = Path("data/realtime_results")
FOUND_DIR_CAM0 = OUT_BASE / "camera0_found"
FOUND_DIR_CAM1 = OUT_BASE / "camera1_found"
REALTIME_FOUND_DIR_CAM0 = OUT_BASE / "camera0_realtime"
REALTIME_FOUND_DIR_CAM1 = OUT_BASE / "camera1_realtime"

# Create output directories
for dir_path in [FOUND_DIR_CAM0, FOUND_DIR_CAM1, REALTIME_FOUND_DIR_CAM0, REALTIME_FOUND_DIR_CAM1]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ---------- HELPERS ----------
def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """
    Workflow outputs can be dict or list. Return first 'predictions'-like list found.
    """
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
        if ONLY_CLASS:
            if pred_class(p).lower() != ONLY_CLASS.lower():
                continue
        out.append(p)
    return out


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess frame to enhance crack visibility.
    Applies CLAHE for contrast enhancement and denoising.
    """
    if not ENABLE_PREPROCESSING:
        return frame
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Convert back to BGR for inference
    enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    # Blend with original (70% enhanced, 30% original) to preserve color info
    blended = cv2.addWeighted(enhanced_bgr, 0.7, frame, 0.3, 0)
    
    return blended


def check_frame_quality(frame: np.ndarray) -> Tuple[bool, float]:
    """
    Check if frame is good quality (not blurry) using Laplacian variance.
    Returns (is_good_quality, blur_score)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_good = laplacian_var >= BLUR_THRESHOLD
    return is_good, laplacian_var


def calculate_crack_area(pred: Dict[str, Any]) -> float:
    """Calculate approximate area of crack detection"""
    w = float(pred.get("width", 0))
    h = float(pred.get("height", 0))
    return w * h


def classify_severity(confidence: float) -> str:
    """Classify crack severity based on confidence"""
    if confidence >= SEVERITY_CRITICAL:
        return "CRITICAL"
    elif confidence >= SEVERITY_HIGH:
        return "HIGH"
    elif confidence >= SEVERITY_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


def draw_detections(frame, preds):
    """Draw bounding boxes and labels on frame with severity indicators"""
    for pred in preds:
        # Get bounding box coordinates
        x = int(pred.get("x", 0))
        y = int(pred.get("y", 0))
        w = int(pred.get("width", 0))
        h = int(pred.get("height", 0))
        
        # Convert center coordinates to top-left coordinates
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        # Determine color based on severity
        conf = pred_conf(pred)
        severity = classify_severity(conf)
        if severity == "CRITICAL":
            color = (0, 0, 255)  # Red
            thickness = 3
        elif severity == "HIGH":
            color = (0, 165, 255)  # Orange
            thickness = 3
        elif severity == "MEDIUM":
            color = (0, 255, 255)  # Yellow
            thickness = 2
        else:
            color = (0, 255, 0)  # Green
            thickness = 2
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with confidence and severity
        area = calculate_crack_area(pred)
        label = f"{pred_class(pred)} [{severity}]: {conf:.2f} ({int(area)}px)"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


# ---------- SHARED STATE (per camera) ----------
class CameraState:
    def __init__(self, camera_id: int, found_dir: Path, realtime_dir: Path):
        self.camera_id = camera_id
        self.found_dir = found_dir
        self.realtime_dir = realtime_dir
        
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        self.latest_result = {"status": "idle", "best": 0.0, "count": 0, "predictions": []}
        self.result_lock = threading.Lock()
        
        # Detection persistence tracking
        self.detection_history = deque(maxlen=PERSISTENCE_FRAMES)
        self.history_lock = threading.Lock()
        
        # Statistics
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
        }
        self.stats_lock = threading.Lock()
        
        self.stop_flag = False
    
    def check_detection_persistence(self) -> bool:
        """Check if detection has persisted across recent frames"""
        if not ENABLE_PERSISTENCE:
            return True
        
        with self.history_lock:
            if len(self.detection_history) < PERSISTENCE_FRAMES:
                return False
            # All recent frames must have detections
            return all(self.detection_history)
    
    def update_detection_history(self, has_detection: bool):
        """Update detection history"""
        with self.history_lock:
            self.detection_history.append(has_detection)


# ---------- INFERENCE WORKER (per camera) ----------
def inference_loop(cam_state: CameraState):
    min_interval = 1.0 / max(INFER_FPS, 0.1)
    last_infer_time = 0.0
    last_saved_found = 0.0

    while not cam_state.stop_flag:
        now = time.time()
        if now - last_infer_time < min_interval:
            time.sleep(0.01)
            continue

        # grab the newest frame
        with cam_state.frame_lock:
            frame = None if cam_state.latest_frame is None else cam_state.latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        with cam_state.stats_lock:
            cam_state.stats["total_frames"] += 1

        # Check frame quality
        is_good_quality, blur_score = check_frame_quality(frame)
        if not is_good_quality:
            with cam_state.stats_lock:
                cam_state.stats["skipped_blurry"] += 1
            # Update detection history as no detection
            cam_state.update_detection_history(False)
            with cam_state.result_lock:
                cam_state.latest_result["status"] = "blurry"
            time.sleep(0.05)
            continue

        last_infer_time = now

        # Preprocess frame for better crack detection
        processed_frame = preprocess_frame(frame)

        ok, jpg = cv2.imencode(".jpg", processed_frame)
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

            # Filter predictions and remove small detections
            preds = filter_preds(extract_predictions(result))
            preds = [p for p in preds if calculate_crack_area(p) >= MIN_CRACK_AREA]
            
            best = max((pred_conf(p) for p in preds), default=0.0)
            count = len(preds)
            found = count > 0

            # Update detection history
            cam_state.update_detection_history(found)

            # Count severity levels
            if found:
                with cam_state.stats_lock:
                    cam_state.stats["detections_found"] += 1
                    for p in preds:
                        severity = classify_severity(pred_conf(p))
                        if severity == "CRITICAL":
                            cam_state.stats["critical_cracks"] += 1
                        elif severity == "HIGH":
                            cam_state.stats["high_cracks"] += 1
                        elif severity == "MEDIUM":
                            cam_state.stats["medium_cracks"] += 1
                        else:
                            cam_state.stats["low_cracks"] += 1

            # update overlay status
            with cam_state.result_lock:
                cam_state.latest_result["status"] = "found" if found else "not_found"
                cam_state.latest_result["best"] = best
                cam_state.latest_result["count"] = count
                cam_state.latest_result["predictions"] = preds
                cam_state.latest_result["blur_score"] = blur_score

            # Save only if detection is persistent
            is_persistent = cam_state.check_detection_persistence()
            
            # categorize + save (cooldown)
            t = time.time()
            name = f"cam{cam_state.camera_id}_{stamp()}_{int(t*1000)}"

            if found and is_persistent and (t - last_saved_found) >= SAVE_COOLDOWN_S:
                last_saved_found = t

                with cam_state.stats_lock:
                    cam_state.stats["total_saved"] += 1

                # Enhanced metadata
                metadata = {
                    "result": result,
                    "blur_score": blur_score,
                    "detections": count,
                    "max_confidence": best,
                    "timestamp": t,
                    "severities": {classify_severity(pred_conf(p)): pred_conf(p) for p in preds},
                    "preprocessing_enabled": ENABLE_PREPROCESSING,
                    "persistence_frames": PERSISTENCE_FRAMES,
                }

                # Save in found
                base_found = cam_state.found_dir / name
                cv2.imwrite(str(base_found) + ".jpg", frame)  # Save original frame
                cv2.imwrite(str(base_found) + "_enhanced.jpg", processed_frame)  # Save enhanced
                Path(str(base_found) + ".json").write_text(json.dumps(metadata, indent=2))

                # ALSO save in realtime_found
                base_rt = cam_state.realtime_dir / name
                cv2.imwrite(str(base_rt) + ".jpg", frame)
                cv2.imwrite(str(base_rt) + "_enhanced.jpg", processed_frame)
                Path(str(base_rt) + ".json").write_text(json.dumps(metadata, indent=2))

        except Exception as e:
            with cam_state.result_lock:
                cam_state.latest_result["status"] = "error"
                cam_state.latest_result["best"] = 0.0
                cam_state.latest_result["count"] = 0
                cam_state.latest_result["predictions"] = []
            time.sleep(0.05)


# ---------- CAMERA CAPTURE THREAD ----------
def camera_capture_thread(cam_state: CameraState, camera_index: int):
    """Dedicated thread for capturing frames from one camera"""
    # For Raspberry Pi 5, use V4L2 backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"ERROR: Camera {camera_index} not available!")
        cam_state.stop_flag = True
        return

    print(f"Camera {cam_state.camera_id} started on /dev/video{camera_index}")

    while not cam_state.stop_flag:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        with cam_state.frame_lock:
            cam_state.latest_frame = frame

    cap.release()


# ---------- MAIN (dual camera display) ----------
def main():
    print("Starting Raspberry Pi 5 Dual Camera Crack Detection")
    print(f"Camera 0: /dev/video{CAMERA_0_INDEX}")
    print(f"Camera 1: /dev/video{CAMERA_1_INDEX}")
    
    # Initialize camera states
    cam0_state = CameraState(0, FOUND_DIR_CAM0, REALTIME_FOUND_DIR_CAM0)
    cam1_state = CameraState(1, FOUND_DIR_CAM1, REALTIME_FOUND_DIR_CAM1)

    # Start capture threads
    capture0_thread = threading.Thread(target=camera_capture_thread, args=(cam0_state, CAMERA_0_INDEX), daemon=True)
    capture1_thread = threading.Thread(target=camera_capture_thread, args=(cam1_state, CAMERA_1_INDEX), daemon=True)
    
    # Start inference threads
    inference0_thread = threading.Thread(target=inference_loop, args=(cam0_state,), daemon=True)
    inference1_thread = threading.Thread(target=inference_loop, args=(cam1_state,), daemon=True)

    capture0_thread.start()
    capture1_thread.start()
    inference0_thread.start()
    inference1_thread.start()

    print("All threads started. Press ESC to quit.")

    # Main display loop
    while True:
        frames_to_show = []
        
        # Get frame from camera 0
        with cam0_state.frame_lock:
            frame0 = cam0_state.latest_frame.copy() if cam0_state.latest_frame is not None else None
        
        # Get frame from camera 1
        with cam1_state.frame_lock:
            frame1 = cam1_state.latest_frame.copy() if cam1_state.latest_frame is not None else None

        if SHOW_PREVIEW:
            # Process camera 0 frame
            if frame0 is not None:
                with cam0_state.result_lock:
                    status0 = cam0_state.latest_result["status"]
                    best0 = cam0_state.latest_result["best"]
                    count0 = cam0_state.latest_result["count"]
                    preds0 = cam0_state.latest_result["predictions"].copy()
                    blur0 = cam0_state.latest_result.get("blur_score", 0)
                
                with cam0_state.stats_lock:
                    stats0 = cam0_state.stats.copy()

                frame0 = draw_detections(frame0, preds0)
                
                # Multi-line status overlay
                y_offset = 25
                overlay0_line1 = f"CAM0 | {status0} | Dets={count0} | Conf={best0:.2f}"
                overlay0_line2 = f"Quality={blur0:.1f} | Saved={stats0['total_saved']} | Critical={stats0['critical_cracks']}"
                
                cv2.putText(frame0, overlay0_line1, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame0, overlay0_line2, (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                frames_to_show.append(frame0)

            # Process camera 1 frame
            if frame1 is not None:
                with cam1_state.result_lock:
                    status1 = cam1_state.latest_result["status"]
                    best1 = cam1_state.latest_result["best"]
                    count1 = cam1_state.latest_result["count"]
                    preds1 = cam1_state.latest_result["predictions"].copy()
                    blur1 = cam1_state.latest_result.get("blur_score", 0)
                
                with cam1_state.stats_lock:
                    stats1 = cam1_state.stats.copy()

                frame1 = draw_detections(frame1, preds1)
                
                # Multi-line status overlay
                y_offset = 25
                overlay1_line1 = f"CAM1 | {status1} | Dets={count1} | Conf={best1:.2f}"
                overlay1_line2 = f"Quality={blur1:.1f} | Saved={stats1['total_saved']} | Critical={stats1['critical_cracks']}"
                
                cv2.putText(frame1, overlay1_line1, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame1, overlay1_line2, (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                frames_to_show.append(frame1)

            # Display frames
            if len(frames_to_show) == 2:
                # Stack frames horizontally
                combined = cv2.hconcat(frames_to_show)
                cv2.imshow("Pi5 Dual Camera - Crack Detection", combined)
            elif len(frames_to_show) == 1:
                cv2.imshow("Pi5 Dual Camera - Crack Detection", frames_to_show[0])

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        time.sleep(0.01)

    # Cleanup
    print("\nStopping cameras...")
    cam0_state.stop_flag = True
    cam1_state.stop_flag = True
    
    time.sleep(0.5)
    cv2.destroyAllWindows()
    
    # Print statistics summary
    print("\n" + "="*60)
    print("DETECTION STATISTICS SUMMARY")
    print("="*60)
    
    for cam_id, cam_state in [(0, cam0_state), (1, cam1_state)]:
        with cam_state.stats_lock:
            stats = cam_state.stats
        print(f"\nCamera {cam_id}:")
        print(f"  Total Frames: {stats['total_frames']}")
        print(f"  Processed: {stats['processed_frames']}")
        print(f"  Skipped (blurry): {stats['skipped_blurry']}")
        print(f"  Detections Found: {stats['detections_found']}")
        print(f"  Saved to Disk: {stats['total_saved']}")
        print(f"  Severity Breakdown:")
        print(f"    - Critical: {stats['critical_cracks']}")
        print(f"    - High: {stats['high_cracks']}")
        print(f"    - Medium: {stats['medium_cracks']}")
        print(f"    - Low: {stats['low_cracks']}")
    
    print("\n" + "="*60)
    print("Done.")


if __name__ == "__main__":
    main()
