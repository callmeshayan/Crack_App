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
from flask import Flask, Response, render_template_string, jsonify, send_file
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage

# Conditional imports for online/offline models
try:
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Warning: inference_sdk not available. Only offline mode will work.")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Only online mode will work.")

# ---------------- ENV ----------------
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# ---------------- INTERACTIVE CONFIGURATION ----------------
def get_operator_input():
    """
    Interactive prompt to get inspection parameters from operator
    """
    print("\n" + "="*60)
    print("AUTOMATED PIPELINE INSPECTION SYSTEM")
    print("System Configuration")
    print("="*60 + "\n")
    
    # Model selection
    print("SELECT DETECTION MODEL:")
    print("  1. On-Device AI Model (YOLOv11n - 68% mAP, Edge Computing)")
    print("  2. Cloud-Based AI Model (Roboflow API - Real-time Processing)")
    
    default_mode = os.getenv("MODEL_MODE", "offline").strip().lower()
    default_choice = "1" if default_mode == "offline" else "2"
    
    while True:
        choice = input(f"\nEnter choice [1/2] (default: {default_choice}): ").strip() or default_choice
        if choice in ["1", "2"]:
            model_mode = "offline" if choice == "1" else "online"
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Pipeline length
    default_length = os.getenv("PIPELINE_LENGTH", "100.0")
    print(f"\nPIPELINE LENGTH:")
    while True:
        length_input = input(f"  Enter pipeline length in meters (default: {default_length}): ").strip() or default_length
        try:
            pipeline_length = float(length_input)
            if pipeline_length > 0:
                break
            print("  Error: Length must be positive.")
        except ValueError:
            print("  Error: Please enter a valid number.")
    
    # Velocity
    default_velocity = os.getenv("ROBOT_VELOCITY", "0.5")
    default_unit = os.getenv("VELOCITY_UNIT", "m/s").lower()
    
    print(f"\nROBOT VELOCITY:")
    print("  Available units: m/s or km/h")
    
    while True:
        unit_input = input(f"  Enter unit [m/s/km/h] (default: {default_unit}): ").strip().lower() or default_unit
        if unit_input in ["m/s", "km/h"]:
            velocity_unit = unit_input
            break
        print("  Error: Please enter 'm/s' or 'km/h'.")
    
    while True:
        velocity_input = input(f"  Enter velocity in {velocity_unit} (default: {default_velocity}): ").strip() or default_velocity
        try:
            velocity = float(velocity_input)
            if velocity > 0:
                break
            print("  Error: Velocity must be positive.")
        except ValueError:
            print("  Error: Please enter a valid number.")
    
    # Summary
    print("\n" + "-"*60)
    print("CONFIGURATION SUMMARY:")
    print("-"*60)
    model_name = "On-Device AI Model (YOLOv11n - 68% mAP)" if model_mode == "offline" else "Cloud-Based AI Model (Roboflow API)"
    print(f"  Detection Model:  {model_name}")
    print(f"  Pipeline Length:  {pipeline_length:.2f} meters")
    print(f"  Robot Velocity:   {velocity:.2f} {velocity_unit}")
    
    # Convert velocity to m/s
    if velocity_unit == "km/h":
        velocity_mps = velocity / 3.6
    else:
        velocity_mps = velocity
    
    estimated_time = pipeline_length / velocity_mps if velocity_mps > 0 else 0
    print(f"  Estimated Time:   {estimated_time:.1f} seconds ({estimated_time/60:.1f} minutes)")
    print("-"*60)
    
    confirm = input("\nProceed with this configuration? [Y/n]: ").strip().lower() or 'y'
    if confirm != 'y':
        print("Configuration cancelled. Exiting...")
        exit(0)
    
    print("\n[INIT] Starting system with operator configuration...\n")
    
    return {
        'model_mode': model_mode,
        'pipeline_length': pipeline_length,
        'velocity': velocity,
        'velocity_unit': velocity_unit,
        'velocity_mps': velocity_mps,
    }

# Get operator input
operator_config = get_operator_input()

# Apply operator configuration
MODEL_MODE = operator_config['model_mode']
PIPELINE_LENGTH_METERS = operator_config['pipeline_length']
ROBOT_VELOCITY = operator_config['velocity']
VELOCITY_UNIT = operator_config['velocity_unit']
ROBOT_VELOCITY_MPS = operator_config['velocity_mps']
ESTIMATED_INSPECTION_DURATION_SEC = PIPELINE_LENGTH_METERS / ROBOT_VELOCITY_MPS if ROBOT_VELOCITY_MPS > 0 else 600.0

# ---------------- MODEL CONFIGURATION ----------------
# Path to your trained YOLO model (for offline mode)
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    "/Users/shayannaghashpour/Desktop/--/pipe_crack_ai/runs/detect/train_20260314_134701/weights/best.pt"
).strip()

# Device for YOLO inference: 'cpu', 'mps' (Mac M1/M2), or '0' (CUDA GPU)
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu").strip()

# Roboflow configuration (for online mode)
API_KEY = os.getenv("RF_API_KEY", "").strip()
WORKSPACE = os.getenv("RF_WORKSPACE", "").strip()
WORKFLOW_ID = os.getenv("RF_WORKFLOW_ID", "").strip()

# Initialize model based on mode
client = None
local_model = None

if MODEL_MODE == "online":
    if not ROBOFLOW_AVAILABLE:
        raise ValueError("Online mode selected but inference_sdk not installed. Install with: pip install inference-sdk")
    if not API_KEY or not WORKSPACE or not WORKFLOW_ID:
        raise ValueError("Online mode requires .env vars: RF_API_KEY, RF_WORKSPACE, RF_WORKFLOW_ID")
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY,
    )
    print(f"[INIT] Cloud-Based AI Model initialized successfully")
elif MODEL_MODE == "offline":
    if not ULTRALYTICS_AVAILABLE:
        raise ValueError("Offline mode selected but ultralytics not installed. Install with: pip install ultralytics")
    if not Path(LOCAL_MODEL_PATH).exists():
        raise ValueError(f"Local model not found at: {LOCAL_MODEL_PATH}")
    # Load model with CPU/MPS for Raspberry Pi compatibility
    local_model = YOLO(LOCAL_MODEL_PATH)
    print(f"[INIT] On-Device AI Model initialized successfully (YOLOv11n - 68% mAP)")
    print(f"[INIT] Model path: {LOCAL_MODEL_PATH}")
    print(f"[INIT] Compute device: {YOLO_DEVICE.upper()}")
    print(f"[INIT] Robot velocity: {ROBOT_VELOCITY} {VELOCITY_UNIT} ({ROBOT_VELOCITY_MPS:.3f} m/s)")
    print(f"[INIT] Pipeline length: {PIPELINE_LENGTH_METERS}m")
    print(f"[INIT] Estimated inspection time: {ESTIMATED_INSPECTION_DURATION_SEC:.1f}s ({ESTIMATED_INSPECTION_DURATION_SEC/60:.1f} min)")
else:
    raise ValueError(f"Invalid MODEL_MODE: {MODEL_MODE}. Must be 'online' or 'offline'")

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


def estimate_crack_position(elapsed_sec: float, velocity_mps: float, pipeline_length_m: float) -> float:
    """
    Estimate crack position along the pipeline based on elapsed time and robot velocity.
    
    IMPORTANT: This is an APPROXIMATE estimate assuming constant robot speed.
    The actual position may vary if the robot speed changes during inspection.
    
    Formula: position_m = velocity_mps * elapsed_time
    
    Args:
        elapsed_sec: Time elapsed since inspection start (seconds)
        velocity_mps: Robot velocity in meters per second
        pipeline_length_m: Total pipeline length in meters
    
    Returns:
        Estimated position in meters from pipe entrance (clamped to pipeline length)
    """
    if not ENABLE_POSITION_TRACKING or velocity_mps <= 0:
        return 0.0
    
    # Calculate position: distance = velocity * time
    position_m = velocity_mps * elapsed_sec
    
    # Clamp to pipeline length
    position_m = max(0.0, min(position_m, pipeline_length_m))
    
    return position_m


def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """Extract predictions from Roboflow workflow result or YOLO result object"""
    # Handle YOLO Results object (offline mode)
    if hasattr(result, 'boxes'):
        predictions = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                box = boxes[i]
                # YOLO format: xyxy, confidence, class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Convert to center format (like Roboflow)
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Get class name
                class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                
                predictions.append({
                    "x": x_center,
                    "y": y_center,
                    "width": width,
                    "height": height,
                    "confidence": conf,
                    "class": class_name,
                    "class_name": class_name,
                })
        return predictions
    
    # Handle Roboflow workflow result (online mode)
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


def draw_location_indicator(frame: np.ndarray, position_m: float, pipeline_length_m: float, 
                            velocity: float, velocity_unit: str) -> np.ndarray:
    """Draw visual location indicator on the frame"""
    h, w = frame.shape[:2]
    
    # Draw info panel background
    panel_height = 120
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    # Title
    cv2.putText(panel, "PIPELINE LOCATION", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Position bar
    bar_x = 10
    bar_y = 40
    bar_width = w - 20
    bar_height = 30
    
    # Background bar
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (80, 80, 80), -1)
    
    # Progress bar
    progress = min(position_m / pipeline_length_m, 1.0) if pipeline_length_m > 0 else 0
    progress_width = int(bar_width * progress)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                 (0, 200, 100), -1)
    
    # Position marker
    marker_x = bar_x + progress_width
    cv2.circle(panel, (marker_x, bar_y + bar_height // 2), 8, (255, 255, 255), -1)
    cv2.circle(panel, (marker_x, bar_y + bar_height // 2), 8, (0, 255, 0), 2)
    
    # Text information
    info_y = bar_y + bar_height + 25
    cv2.putText(panel, f"Position: {position_m:.2f}m / {pipeline_length_m:.1f}m", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, f"Velocity: {velocity:.2f} {velocity_unit}", 
                (w - 200, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Combine panel with frame
    result = np.vstack([panel, frame])
    return result


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


# ---------------- PDF REPORT GENERATION ----------------
def generate_pdf_report(output_path: Path, inspection_data: Dict[str, Any]) -> bool:
    """
    Generate a professional PDF report of the pipeline inspection
    
    Args:
        output_path: Path where PDF will be saved
        inspection_data: Dictionary containing inspection results and metadata
    
    Returns:
        True if successful, False otherwise
    """
    try:
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#16213e'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("Automated Pipeline Inspection Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Inspection Summary
        story.append(Paragraph("Inspection Summary", heading_style))
        
        summary_data = [
            ['Parameter', 'Value'],
            ['Inspection Date', inspection_data.get('date', 'N/A')],
            ['Pipeline Length', f"{inspection_data.get('pipeline_length', 0):.1f} m"],
            ['Robot Velocity', f"{inspection_data.get('velocity', 0):.2f} {inspection_data.get('velocity_unit', 'm/s')}"],
            ['Inspection Duration', f"{inspection_data.get('duration', 0):.1f} sec ({inspection_data.get('duration', 0)/60:.1f} min)"],
            ['Model Mode', inspection_data.get('model_mode', 'N/A').upper()],
            ['Total Frames Processed', str(inspection_data.get('total_frames', 0))],
            ['Total Cracks Detected', str(inspection_data.get('total_cracks', 0))],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Severity Breakdown
        story.append(Paragraph("Severity Breakdown", heading_style))
        
        severity_data = [
            ['Severity Level', 'Count', 'Percentage'],
            ['CRITICAL', str(inspection_data.get('critical_count', 0)), 
             f"{inspection_data.get('critical_count', 0) / max(inspection_data.get('total_cracks', 1), 1) * 100:.1f}%"],
            ['HIGH', str(inspection_data.get('high_count', 0)),
             f"{inspection_data.get('high_count', 0) / max(inspection_data.get('total_cracks', 1), 1) * 100:.1f}%"],
            ['MEDIUM', str(inspection_data.get('medium_count', 0)),
             f"{inspection_data.get('medium_count', 0) / max(inspection_data.get('total_cracks', 1), 1) * 100:.1f}%"],
            ['LOW', str(inspection_data.get('low_count', 0)),
             f"{inspection_data.get('low_count', 0) / max(inspection_data.get('total_cracks', 1), 1) * 100:.1f}%"],
        ]
        
        severity_table = Table(severity_data, colWidths=[2*inch, 2*inch, 2*inch])
        severity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#ff4444')),
            ('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#ff8800')),
            ('BACKGROUND', (0, 3), (0, 3), colors.HexColor('#ffcc00')),
            ('BACKGROUND', (0, 4), (0, 4), colors.HexColor('#44ff44')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(severity_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed Crack List
        if inspection_data.get('cracks', []):
            story.append(PageBreak())
            story.append(Paragraph("Detailed Crack Locations", heading_style))
            
            crack_data = [['ID', 'Position (m)', 'Confidence', 'Severity', 'Camera']]
            for crack in inspection_data.get('cracks', [])[:50]:  # Limit to first 50
                crack_data.append([
                    str(crack.get('crack_id', 'N/A')),
                    f"{crack.get('position_m', 0):.2f}",
                    f"{crack.get('confidence', 0):.2f}",
                    crack.get('severity', 'N/A'),
                    f"CAM{crack.get('camera_id', 0)}",
                ])
            
            crack_table = Table(crack_data, colWidths=[0.8*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.2*inch])
            crack_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(crack_table)
        
        # Build PDF
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"[PDF] Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        
        # Crack history for web visualization
        self.crack_history = []  # List of crack detection records
        self.history_max_size = 100  # Keep last 100 cracks
        
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
        return estimate_crack_position(elapsed, ROBOT_VELOCITY_MPS, PIPELINE_LENGTH_METERS)
    
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
    
    def add_crack_to_history(self, crack_data: Dict[str, Any]):
        """Add crack to history for web visualization"""
        with self.position_lock:
            self.crack_history.append(crack_data)
            # Keep only recent cracks
            if len(self.crack_history) > self.history_max_size:
                self.crack_history.pop(0)
    
    def get_crack_history(self) -> List[Dict[str, Any]]:
        """Get crack history for web API"""
        with self.position_lock:
            return list(self.crack_history)
    
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

        try:
            if MODEL_MODE == "online":
                # Online Roboflow inference
                ok, jpg = cv2.imencode(".jpg", processed)
                if not ok:
                    continue
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
                    f.write(jpg.tobytes())
                    f.flush()

                    result = client.run_workflow(
                        workspace_name=WORKSPACE,
                        workflow_id=WORKFLOW_ID,
                        images={"image": f.name},
                        use_cache=True,
                    )
            else:
                # Offline local YOLO inference
                # Run inference on preprocessed frame
                # YOLO already filters by conf threshold, so we pass it directly
                results = local_model.predict(
                    source=processed,
                    conf=CONF_THRESH,
                    verbose=False,
                    device=YOLO_DEVICE,
                    half=False,  # Use FP32 for better compatibility
                    imgsz=640,  # Match training size
                )
                # Get first result (single image)
                result = results[0] if results and len(results) > 0 else None
                if result is None:
                    print(f"[CAM{cam_state.camera_id}] Warning: YOLO returned no results")
                    continue

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
                    
                    # Add location indicator to annotated images
                    annotated_raw_with_loc = draw_location_indicator(
                        annotated_raw, position_m, PIPELINE_LENGTH_METERS, 
                        ROBOT_VELOCITY, VELOCITY_UNIT
                    )
                    annotated_enhanced_with_loc = draw_location_indicator(
                        annotated_enhanced, position_m, PIPELINE_LENGTH_METERS,
                        ROBOT_VELOCITY, VELOCITY_UNIT
                    )

                    # Prepare serializable result data
                    if MODEL_MODE == "online":
                        result_data = result
                    else:
                        # For YOLO results, extract only serializable data
                        result_data = {
                            "model_type": "yolo",
                            "inference_time_ms": getattr(result, 'speed', {}).get('inference', 0) if hasattr(result, 'speed') else 0,
                            "predictions": [pred],  # Already contains the specific prediction
                        }
                    
                    metadata = {
                        "crack_id": crack_id,
                        "camera_id": cam_state.camera_id,
                        "elapsed_sec": elapsed,
                        "position_m": position_m,
                        "pipeline_length_m": PIPELINE_LENGTH_METERS,
                        "result": result_data,
                        "blur_score": blur_score,
                        "detections": 1,
                        "confidence": conf,
                        "severity": sev,
                        "area_px": area,
                        "class": class_name,
                        "timestamp": t,
                        "model_mode": MODEL_MODE,
                        "preprocessing_enabled": ENABLE_PREPROCESSING,
                        "persistence_enabled": ENABLE_PERSISTENCE,
                        "persistence_frames": PERSISTENCE_FRAMES,
                        "position_tracking_enabled": ENABLE_POSITION_TRACKING,
                    }

                    base_found = cam_state.found_dir / name
                    cv2.imwrite(str(base_found) + ".jpg", frame)
                    cv2.imwrite(str(base_found) + "_enhanced.jpg", processed)
                    cv2.imwrite(str(base_found) + "_annotated.jpg", annotated_raw_with_loc)
                    cv2.imwrite(str(base_found) + "_enhanced_annotated.jpg", annotated_enhanced_with_loc)
                    Path(str(base_found) + ".json").write_text(json.dumps(metadata, indent=2, default=str))

                    base_rt = cam_state.realtime_dir / name
                    cv2.imwrite(str(base_rt) + ".jpg", frame)
                    cv2.imwrite(str(base_rt) + "_enhanced.jpg", processed)
                    cv2.imwrite(str(base_rt) + "_annotated.jpg", annotated_raw_with_loc)
                    cv2.imwrite(str(base_rt) + "_enhanced_annotated.jpg", annotated_enhanced_with_loc)
                    Path(str(base_rt) + ".json").write_text(json.dumps(metadata, indent=2, default=str))
                    
                    # Write to CSV
                    if ENABLE_POSITION_TRACKING:
                        cam_state.write_crack_to_csv(
                            crack_id, elapsed, position_m, conf, sev, 
                            area, class_name, name + ".jpg", t
                        )
                    
                    # Add to web visualization history
                    cam_state.add_crack_to_history({
                        "crack_id": crack_id,
                        "camera_id": cam_state.camera_id,
                        "position_m": position_m,
                        "elapsed_sec": elapsed,
                        "confidence": conf,
                        "severity": sev,
                        "area_px": area,
                        "class": class_name,
                        "timestamp": t,
                        "image_path": str(base_found) + "_annotated.jpg",
                        "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
                    })

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
    <title>Automated Pipeline Inspection System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            color: #00ff88;
            font-size: 2.5em;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
            margin-bottom: 10px;
        }
        .model-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.5em;
            font-weight: bold;
            margin-left: 15px;
            vertical-align: middle;
        }
        .model-badge.offline {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .model-badge.online {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        }
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.7;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #00ff88;
        }
        
        /* Pipeline Visualization */
        .pipeline-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .pipeline-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #00ff88;
        }
        .pipeline-container {
            position: relative;
            width: 100%;
            height: 120px;
            background: linear-gradient(to right, #2a2a2a 0%, #3a3a3a 50%, #2a2a2a 100%);
            border-radius: 60px;
            overflow: visible;
            border: 3px solid #555;
            box-shadow: inset 0 4px 10px rgba(0,0,0,0.5);
        }
        .pipeline-markers {
            position: absolute;
            bottom: -30px;
            width: 100%;
            display: flex;
            justify-content: space-between;
            padding: 0 10px;
            font-size: 0.8em;
            color: #888;
        }
        .crack-marker {
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            cursor: pointer;
            top: 50%;
            transform: translate(-50%, -50%);
            transition: all 0.3s ease;
            z-index: 10;
            animation: pulse 2s infinite;
        }
        .crack-marker.CRITICAL {
            background: #ff0000;
            box-shadow: 0 0 15px #ff0000, 0 0 30px #ff000080;
        }
        .crack-marker.HIGH {
            background: #ff6600;
            box-shadow: 0 0 15px #ff6600, 0 0 30px #ff660080;
        }
        .crack-marker.MEDIUM {
            background: #ffff00;
            box-shadow: 0 0 15px #ffff00, 0 0 30px #ffff0080;
        }
        .crack-marker.LOW {
            background: #00ff00;
            box-shadow: 0 0 15px #00ff00, 0 0 30px #00ff0080;
        }
        .crack-marker:hover {
            transform: translate(-50%, -50%) scale(1.5);
            z-index: 100;
        }
        @keyframes pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.2); }
        }
        
        /* Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(5px);
        }
        .modal-content {
            background: linear-gradient(135deg, #2a2a3e 0%, #1e2742 100%);
            margin: 5% auto;
            padding: 0;
            border-radius: 15px;
            width: 90%;
            max-width: 800px;
            box-shadow: 0 10px 50px rgba(0, 255, 136, 0.3);
            border: 2px solid rgba(0, 255, 136, 0.3);
            max-height: 90vh;
            overflow-y: auto;
        }
        .modal-header {
            padding: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-body {
            padding: 20px;
        }
        .close {
            color: #aaa;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }
        .close:hover {
            color: #00ff88;
        }
        .crack-image {
            width: 100%;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px solid rgba(255, 255, 255, 0.2);
        }
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .detail-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #00ff88;
        }
        .detail-label {
            font-size: 0.9em;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        .detail-value {
            font-size: 1.3em;
            font-weight: bold;
        }
        .severity-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .severity-CRITICAL {
            background: #ff0000;
            color: #fff;
        }
        .severity-HIGH {
            background: #ff6600;
            color: #fff;
        }
        .severity-MEDIUM {
            background: #ffff00;
            color: #000;
        }
        .severity-LOW {
            background: #00ff00;
            color: #000;
        }
        
        /* Camera Feed */
        .camera-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .camera-box {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .camera-box h2 {
            color: #00ff88;
            margin-bottom: 15px;
        }
        .camera-box img {
            width: 100%;
            border-radius: 10px;
            border: 2px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Legend */
        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 50%;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            h1 {
                font-size: 1.8em;
            }
            .pipeline-container {
                height: 80px;
            }
            .modal-content {
                width: 95%;
                margin: 10% auto;
            }
        }
        
        /* Unit Selectors */
        .unit-controls {
            display:flex; gap:15px; justify-content:center;
            margin-top:15px; flex-wrap:wrap;
            padding:10px;
        }
        .unit-group {
            display:flex; flex-direction:column; align-items:center;
        }
        .unit-label {
            font-size:0.85em; color:#667eea; margin-bottom:5px;
            opacity:0.9; font-weight:bold;
        }
        .unit-selector {
            padding:8px 15px; border-radius:8px;
            border:2px solid #667eea; background:white;
            color:#667eea; font-size:0.9em; font-weight:bold;
            cursor:pointer; transition:all 0.3s;
        }
        .unit-selector:hover {
            background:#667eea;
            color:white;
            box-shadow:0 4px 15px rgba(102, 126, 234, 0.4);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            Automated Pipeline Inspection System
            <span class="model-badge {{ model_mode }}">{{ model_mode_display }}</span>
        </h1>
        <div class="unit-controls">
            <div class="unit-group">
                <div class="unit-label">Length Unit</div>
                <select class="unit-selector" id="length-unit" onchange="updateUnits()">
                    <option value="m" selected>Meters (m)</option>
                    <option value="cm">Centimeters (cm)</option>
                    <option value="km">Kilometers (km)</option>
                </select>
            </div>
            <div class="unit-group">
                <div class="unit-label">Velocity Unit</div>
                <select class="unit-selector" id="velocity-unit" onchange="updateUnits()">
                    <option value="m/s" selected>m/s</option>
                    <option value="mm/s">mm/s</option>
                    <option value="km/h">km/h</option>
                </select>
            </div>
        </div>
    </div>
    
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-label">Pipeline Length</div>
            <div class="stat-value" id="pipeline-length">-</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Current Position</div>
            <div class="stat-value" id="current-position">-</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Velocity</div>
            <div class="stat-value" id="velocity">-</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Total Cracks</div>
            <div class="stat-value" id="total-cracks">0</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Critical</div>
            <div class="stat-value" style="color: #ff0000;" id="critical-count">0</div>
        </div>
        <div class="stat-item" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <button onclick="generatePDF()" style="background: none; border: 2px solid white; color: white; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 1em; font-weight: bold;">
                Generate PDF Report
            </button>
        </div>
    </div>
    
    <div class="pipeline-section">
        <div class="pipeline-title">Pipeline Visualization</div>
        <div class="pipeline-container" id="pipeline-container">
            <div class="pipeline-markers">
                <span>0m</span>
                <span id="pipeline-end">100m</span>
            </div>
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #ff0000;"></div>
                <span>Critical</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ff6600;"></div>
                <span>High</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ffff00;"></div>
                <span>Medium</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #00ff00;"></div>
                <span>Low</span>
            </div>
        </div>
    </div>
    
    <div class="camera-container">
        <div class="camera-box">
            <h2>📹 Camera Feed</h2>
            <img src="/video_feed/0" alt="Camera 0">
        </div>
    </div>
    
    <!-- Crack Detail Modal -->
    <div id="crackModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modal-title">Crack Details</h2>
                <span class="close">&times;</span>
            </div>
            <div class="modal-body">
                <img id="modal-image" class="crack-image" src="" alt="Crack Image">
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-label">Crack ID</div>
                        <div class="detail-value" id="modal-crack-id">-</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Position</div>
                        <div class="detail-value" id="modal-position">-</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Confidence</div>
                        <div class="detail-value" id="modal-confidence">-</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Severity</div>
                        <div class="detail-value" id="modal-severity">-</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Area (pixels)</div>
                        <div class="detail-value" id="modal-area">-</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Detection Time</div>
                        <div class="detail-value" id="modal-time">-</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const modal = document.getElementById('crackModal');
        const closeBtn = document.getElementsByClassName('close')[0];
        const pipelineContainer = document.getElementById('pipeline-container');
        
        // Unit conversion functions
        function convertLength(meters, unit) {
            switch(unit) {
                case 'cm': return meters * 100;
                case 'km': return meters / 1000;
                default: return meters; // m
            }
        }

        function convertVelocity(metersPerSec, unit) {
            switch(unit) {
                case 'mm/s': return metersPerSec * 1000;
                case 'km/h': return metersPerSec * 3.6;
                default: return metersPerSec; // m/s
            }
        }

        function getLengthUnit() {
            return document.getElementById('length-unit').value;
        }

        function getVelocityUnit() {
            return document.getElementById('velocity-unit').value;
        }

        function updateUnits() {
            // Re-fetch and update display with new units
            updatePipeline();
        }
        
        closeBtn.onclick = function() {
            modal.style.display = 'none';
        }
        
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
        
        function showCrackDetail(crack) {
            const lengthUnit = getLengthUnit();
            const convertedPosition = convertLength(crack.position_m, lengthUnit);
            
            document.getElementById('modal-title').textContent = `Crack #${crack.crack_id} Details`;
            document.getElementById('modal-image').src = `/crack_image/${crack.camera_id}/${crack.crack_id}`;
            document.getElementById('modal-crack-id').textContent = crack.crack_id;
            document.getElementById('modal-position').textContent = convertedPosition.toFixed(lengthUnit === 'km' ? 3 : 2) + ' ' + lengthUnit;
            document.getElementById('modal-confidence').textContent = (crack.confidence * 100).toFixed(1) + '%';
            document.getElementById('modal-area').textContent = Math.round(crack.area_px) + ' px²';
            document.getElementById('modal-time').textContent = crack.timestamp_str;
            
            const severityEl = document.getElementById('modal-severity');
            severityEl.innerHTML = `<span class="severity-badge severity-${crack.severity}">${crack.severity}</span>`;
            
            modal.style.display = 'block';
        }
        
        function updatePipeline() {
            fetch('/api/cracks')
                .then(response => response.json())
                .then(data => {
                    const pipelineLength = data.pipeline_length_m;
                    const cracks = data.cracks;
                    const lengthUnit = getLengthUnit();
                    const velocityUnit = getVelocityUnit();
                    
                    // Convert values
                    const convertedLength = convertLength(pipelineLength, lengthUnit);
                    const convertedPosition = convertLength(data.current_position_m, lengthUnit);
                    const velocityMPS = parseFloat(data.velocity) || 0;
                    const convertedVelocity = convertVelocity(velocityMPS, velocityUnit);
                    
                    // Update stats
                    document.getElementById('pipeline-length').textContent = convertedLength.toFixed(lengthUnit === 'km' ? 3 : lengthUnit === 'cm' ? 0 : 1) + lengthUnit;
                    document.getElementById('pipeline-end').textContent = convertedLength.toFixed(0) + lengthUnit;
                    document.getElementById('current-position').textContent = convertedPosition.toFixed(lengthUnit === 'km' ? 3 : 2) + lengthUnit;
                    document.getElementById('velocity').textContent = convertedVelocity.toFixed(velocityUnit === 'km/h' ? 2 : velocityUnit === 'mm/s' ? 0 : 2) + ' ' + velocityUnit;
                    document.getElementById('total-cracks').textContent = cracks.length;
                    
                    const criticalCount = cracks.filter(c => c.severity === 'CRITICAL').length;
                    document.getElementById('critical-count').textContent = criticalCount;
                    
                    // Clear existing markers
                    const existingMarkers = pipelineContainer.querySelectorAll('.crack-marker');
                    existingMarkers.forEach(m => m.remove());
                    
                    // Add crack markers
                    cracks.forEach(crack => {
                        const marker = document.createElement('div');
                        marker.className = `crack-marker ${crack.severity}`;
                        const percentage = (crack.position_m / pipelineLength) * 100;
                        marker.style.left = percentage + '%';
                        marker.title = `Crack #${crack.crack_id} at ${crack.position_m.toFixed(2)}m - ${crack.severity}`;
                        marker.onclick = () => showCrackDetail(crack);
                        pipelineContainer.appendChild(marker);
                    });
                })
                .catch(error => console.error('Error fetching crack data:', error));
        }
        
        function generatePDF() {
            const button = event.target;
            button.textContent = 'Generating...';
            button.disabled = true;
            
            fetch('/generate_report')
                .then(response => response.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'pipeline_inspection_report.pdf';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    button.textContent = 'Generate PDF Report';
                    button.disabled = false;
                    alert('PDF report generated successfully!');
                })
                .catch(error => {
                    console.error('Error generating PDF:', error);
                    alert('Error generating PDF report. Check console for details.');
                    button.textContent = 'Generate PDF Report';
                    button.disabled = false;
                });
        }
        
        // Update every 2 seconds
        updatePipeline();
        setInterval(updatePipeline, 2000);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    model_mode_display = "Local YOLO11n (68%)" if MODEL_MODE == "offline" else "Roboflow API"
    return render_template_string(
        HTML_TEMPLATE,
        model_mode=MODEL_MODE,
        model_mode_display=model_mode_display
    )


@app.route('/api/cracks')
def get_cracks():
    """API endpoint to get all detected cracks"""
    if cam0 is None:
        return jsonify({"error": "Camera not initialized"}), 500
    
    cracks = cam0.get_crack_history()
    current_position = cam0.get_estimated_position()
    
    return jsonify({
        "pipeline_length_m": PIPELINE_LENGTH_METERS,
        "current_position_m": current_position,
        "velocity": ROBOT_VELOCITY,
        "velocity_unit": VELOCITY_UNIT,
        "total_cracks": len(cracks),
        "cracks": cracks,
    })


@app.route('/generate_report')
def generate_report():
    """Generate PDF report of inspection"""
    try:
        if cam0 is None:
            return "Camera not initialized", 500
        
        # Collect inspection data
        cracks = cam0.get_crack_history()
        elapsed = cam0.get_elapsed_time()
        
        # Count by severity
        critical = sum(1 for c in cracks if c.get('severity') == 'CRITICAL')
        high = sum(1 for c in cracks if c.get('severity') == 'HIGH')
        medium = sum(1 for c in cracks if c.get('severity') == 'MEDIUM')
        low = sum(1 for c in cracks if c.get('severity') == 'LOW')
        
        with cam0.stats_lock:
            total_frames = cam0.stats.get('total_frames', 0)
        
        inspection_data = {
            'date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'pipeline_length': PIPELINE_LENGTH_METERS,
            'velocity': ROBOT_VELOCITY,
            'velocity_unit': VELOCITY_UNIT,
            'duration': elapsed,
            'model_mode': MODEL_MODE,
            'total_frames': total_frames,
            'total_cracks': len(cracks),
            'critical_count': critical,
            'high_count': high,
            'medium_count': medium,
            'low_count': low,
            'cracks': cracks,
        }
        
        # Generate PDF
        pdf_path = REPORTS_DIR / f"inspection_report_{stamp()}.pdf"
        success = generate_pdf_report(pdf_path, inspection_data)
        
        if success and pdf_path.exists():
            return send_file(pdf_path, 
                           mimetype='application/pdf',
                           as_attachment=True,
                           download_name='pipeline_inspection_report.pdf')
        else:
            return "Error generating PDF report", 500
            
    except Exception as e:
        print(f"[PDF] Error in generate_report: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500


@app.route('/crack_image/<int:camera_id>/<int:crack_id>')
def get_crack_image(camera_id, crack_id):
    """Serve crack image by ID"""
    if cam0 is None:
        return "Camera not initialized", 500
    
    cracks = cam0.get_crack_history()
    crack = next((c for c in cracks if c["crack_id"] == crack_id), None)
    
    if crack and Path(crack["image_path"]).exists():
        return send_file(crack["image_path"], mimetype='image/jpeg')
    else:
        return "Image not found", 404


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

