"""
Pipeline Crack Detection with Location Estimation

This script processes a pipeline inspection video and estimates crack locations
based on video progress and known pipeline length.

ASSUMPTIONS:
- Single camera recording
- Approximately constant robot speed
- Known total pipeline length
- Robot records from pipe entrance to exit
- Speed value is UNKNOWN

LOCATION ESTIMATION:
The crack position is estimated using simple proportional calculation:
    position_m = pipeline_length_m * (current_frame / total_frames)

This is an APPROXIMATE estimate based on frame progress, NOT precise odometry.
"""

import os
import csv
import time
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
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

# ---------- PIPELINE SETTINGS ----------
# USER: Set your pipeline length in meters
PIPELINE_LENGTH_METERS = 100.0  # Change this to your actual pipeline length

# VIDEO INPUT
VIDEO_PATH = "path/to/your/inspection_video.mp4"  # Change this to your video path

# ---------- DETECTION SETTINGS ----------
CONF_THRESH = float(os.getenv("RF_CONF", "0.5"))
INFER_FPS = 2.0  # Process 2 frames per second
ONLY_CLASS = "crack"
SHOW_PREVIEW = True

# Enhanced detection settings
ENABLE_PREPROCESSING = True
ENABLE_PERSISTENCE = True
PERSISTENCE_FRAMES = 3
BLUR_THRESHOLD = 100.0
MIN_CRACK_AREA = 100

# Severity classification
SEVERITY_CRITICAL = 0.85
SEVERITY_HIGH = 0.70
SEVERITY_MEDIUM = 0.55

# Output directories
OUT_BASE = Path("data/pipeline_inspection")
FOUND_DIR = OUT_BASE / "detected_cracks"
ANNOTATED_DIR = OUT_BASE / "annotated_frames"
REPORTS_DIR = OUT_BASE / "reports"

for p in [FOUND_DIR, ANNOTATED_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# CSV report file
CRACK_REPORT_CSV = REPORTS_DIR / f"crack_report_{time.strftime('%Y%m%d_%H%M%S')}.csv"
CRACK_REPORT_JSON = REPORTS_DIR / f"crack_report_{time.strftime('%Y%m%d_%H%M%S')}.json"


# ---------- HELPER FUNCTIONS ----------
def estimate_crack_position(frame_index: int, total_frames: int, pipeline_length_m: float) -> float:
    """
    Estimate crack position along the pipeline based on video progress.
    
    IMPORTANT: This is an APPROXIMATE estimate assuming constant robot speed.
    The actual position may vary if the robot speed changes during inspection.
    
    Args:
        frame_index: Current frame number (0-based)
        total_frames: Total number of frames in video
        pipeline_length_m: Total pipeline length in meters
    
    Returns:
        Estimated position in meters from pipe entrance
    """
    if total_frames == 0:
        return 0.0
    
    # Ensure frame_index is within valid range
    frame_index = max(0, min(frame_index, total_frames - 1))
    
    # Simple proportional calculation
    # position_m = pipeline_length_m * (frame_index / total_frames)
    position_m = pipeline_length_m * (frame_index / total_frames)
    
    return position_m


def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """Extract predictions from workflow result"""
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
    """Enhance frame for better crack detection"""
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
    """Check if frame quality is good enough for detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (lap_var >= BLUR_THRESHOLD), float(lap_var)


def calculate_crack_area(pred: Dict[str, Any]) -> float:
    """Calculate approximate crack area in pixels"""
    w = float(pred.get("width", 0))
    h = float(pred.get("height", 0))
    return w * h


def classify_severity(confidence: float) -> str:
    """Classify crack severity based on confidence"""
    if confidence >= SEVERITY_CRITICAL:
        return "CRITICAL"
    if confidence >= SEVERITY_HIGH:
        return "HIGH"
    if confidence >= SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def draw_detections_with_position(frame: np.ndarray, preds: List[Dict[str, Any]], position_m: float) -> np.ndarray:
    """Draw bounding boxes with severity indicators and position information"""
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
        
        # Color based on severity
        if severity == "CRITICAL":
            color, thickness = (0, 0, 255), 3
        elif severity == "HIGH":
            color, thickness = (0, 165, 255), 3
        elif severity == "MEDIUM":
            color, thickness = (0, 255, 255), 2
        else:
            color, thickness = (0, 255, 0), 2
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with position
        area = calculate_crack_area(pred)
        label = f"Crack at {position_m:.2f}m [{severity}] {conf:.2f}"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def process_video():
    """Main video processing function"""
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        raise ValueError("Video has 0 frames or cannot determine frame count")
    
    print(f"Video: {VIDEO_PATH}")
    print(f"Total frames: {total_frames}")
    print(f"Video FPS: {video_fps}")
    print(f"Pipeline length: {PIPELINE_LENGTH_METERS}m")
    print(f"Processing every ~{1/INFER_FPS:.2f} frames (target {INFER_FPS} FPS)")
    print("="*60)
    
    # Initialize tracking
    frame_interval = max(1, int(video_fps / INFER_FPS))
    detection_history = deque(maxlen=PERSISTENCE_FRAMES)
    crack_records = []
    crack_counter = 0
    
    # Statistics
    stats = {
        "total_frames": total_frames,
        "processed_frames": 0,
        "skipped_blurry": 0,
        "detections_found": 0,
        "cracks_saved": 0,
        "critical_cracks": 0,
        "high_cracks": 0,
        "medium_cracks": 0,
        "low_cracks": 0,
    }
    
    # CSV file setup
    csv_file = open(CRACK_REPORT_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "crack_id", "frame_index", "time_sec", "position_m", 
        "confidence", "severity", "area_px", "image_name", "enhanced_image_name"
    ])
    
    frame_idx = 0
    last_process_frame = -frame_interval
    
    print("Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate current position (even if we don't process this frame)
        position_m = estimate_crack_position(frame_idx, total_frames, PIPELINE_LENGTH_METERS)
        time_sec = frame_idx / video_fps if video_fps > 0 else 0
        
        # Process only at target FPS
        if frame_idx - last_process_frame < frame_interval:
            frame_idx += 1
            continue
        
        last_process_frame = frame_idx
        stats["processed_frames"] += 1
        
        # Check frame quality
        is_good_quality, blur_score = check_frame_quality(frame)
        if not is_good_quality:
            stats["skipped_blurry"] += 1
            detection_history.append(False)
            frame_idx += 1
            continue
        
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        
        # Encode and run inference
        ok, jpg = cv2.imencode(".jpg", processed_frame)
        if not ok:
            frame_idx += 1
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
            
            # Filter predictions
            preds = filter_preds(extract_predictions(result))
            preds = [p for p in preds if calculate_crack_area(p) >= MIN_CRACK_AREA]
            
            found = len(preds) > 0
            detection_history.append(found)
            
            if found:
                stats["detections_found"] += 1
            
            # Check persistence
            is_persistent = (not ENABLE_PERSISTENCE) or (
                len(detection_history) >= PERSISTENCE_FRAMES and all(detection_history)
            )
            
            # Save if persistent detection
            if found and is_persistent:
                # Process each crack separately
                for pred_idx, pred in enumerate(preds):
                    crack_counter += 1
                    conf = pred_conf(pred)
                    severity = classify_severity(conf)
                    area = calculate_crack_area(pred)
                    
                    # Update severity stats
                    if severity == "CRITICAL":
                        stats["critical_cracks"] += 1
                    elif severity == "HIGH":
                        stats["high_cracks"] += 1
                    elif severity == "MEDIUM":
                        stats["medium_cracks"] += 1
                    else:
                        stats["low_cracks"] += 1
                    
                    # Generate filenames
                    base_name = f"crack_{crack_counter:04d}_frame_{frame_idx:06d}_pos_{position_m:.2f}m"
                    img_name = f"{base_name}.jpg"
                    enhanced_name = f"{base_name}_enhanced.jpg"
                    json_name = f"{base_name}.json"
                    
                    # Draw detections on frame
                    annotated_frame = draw_detections_with_position(frame.copy(), [pred], position_m)
                    
                    # Save images
                    cv2.imwrite(str(FOUND_DIR / img_name), frame)
                    cv2.imwrite(str(FOUND_DIR / enhanced_name), processed_frame)
                    cv2.imwrite(str(ANNOTATED_DIR / img_name), annotated_frame)
                    
                    # Save metadata
                    metadata = {
                        "crack_id": crack_counter,
                        "frame_index": frame_idx,
                        "time_sec": time_sec,
                        "position_m": position_m,
                        "confidence": conf,
                        "severity": severity,
                        "area_px": area,
                        "blur_score": blur_score,
                        "prediction": pred,
                        "full_result": result,
                    }
                    
                    with open(FOUND_DIR / json_name, 'w') as jf:
                        json.dump(metadata, jf, indent=2)
                    
                    # Write to CSV
                    csv_writer.writerow([
                        crack_counter, frame_idx, f"{time_sec:.2f}", f"{position_m:.2f}",
                        f"{conf:.3f}", severity, int(area), img_name, enhanced_name
                    ])
                    
                    # Store record for JSON summary
                    crack_records.append(metadata)
                    
                    stats["cracks_saved"] += 1
                    
                    print(f"✓ Crack #{crack_counter} at {position_m:.2f}m (frame {frame_idx}, {severity}, conf={conf:.3f})")
            
            # Show preview if enabled
            if SHOW_PREVIEW and found:
                preview = draw_detections_with_position(frame.copy(), preds, position_m)
                status = f"Frame {frame_idx}/{total_frames} | Pos: {position_m:.2f}m | Cracks: {len(preds)}"
                cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Pipeline Inspection", preview)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break
        
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    
    # Save JSON summary
    summary = {
        "video_path": VIDEO_PATH,
        "pipeline_length_m": PIPELINE_LENGTH_METERS,
        "statistics": stats,
        "cracks": crack_records,
        "processing_settings": {
            "conf_thresh": CONF_THRESH,
            "infer_fps": INFER_FPS,
            "preprocessing": ENABLE_PREPROCESSING,
            "persistence_frames": PERSISTENCE_FRAMES,
            "blur_threshold": BLUR_THRESHOLD,
            "min_crack_area": MIN_CRACK_AREA,
        }
    }
    
    with open(CRACK_REPORT_JSON, 'w') as jf:
        json.dump(summary, jf, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE INSPECTION COMPLETE")
    print("="*60)
    print(f"Pipeline Length: {PIPELINE_LENGTH_METERS}m")
    print(f"Total Frames: {stats['total_frames']}")
    print(f"Processed Frames: {stats['processed_frames']}")
    print(f"Skipped (Blurry): {stats['skipped_blurry']}")
    print(f"Detections Found: {stats['detections_found']}")
    print(f"Cracks Saved: {stats['cracks_saved']}")
    print(f"\nSeverity Breakdown:")
    print(f"  Critical: {stats['critical_cracks']}")
    print(f"  High: {stats['high_cracks']}")
    print(f"  Medium: {stats['medium_cracks']}")
    print(f"  Low: {stats['low_cracks']}")
    print(f"\nReports saved to:")
    print(f"  CSV: {CRACK_REPORT_CSV}")
    print(f"  JSON: {CRACK_REPORT_JSON}")
    print(f"  Images: {FOUND_DIR}")
    print("="*60)


if __name__ == "__main__":
    # Validate inputs
    if not Path(VIDEO_PATH).exists():
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        print("Please set VIDEO_PATH to your inspection video.")
        exit(1)
    
    if PIPELINE_LENGTH_METERS <= 0:
        print("ERROR: PIPELINE_LENGTH_METERS must be > 0")
        exit(1)
    
    print("\n" + "="*60)
    print("PIPELINE CRACK DETECTION WITH LOCATION ESTIMATION")
    print("="*60)
    print("\nIMPORTANT: Location estimates are based on video frame progress")
    print("assuming approximately constant robot speed. This is NOT precise")
    print("odometry and should be verified with physical measurements.\n")
    
    process_video()
