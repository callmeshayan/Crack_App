"""
Raspberry Pi 4 Optimized Dual/Single Camera Real-time Crack Detection with Flask Web Streaming
- Optimized for Raspberry Pi 4 with lower memory and processing power
- Supports single camera fallback mode
- Reduced resolution and frame skipping for better performance
- Access at http://raspberrypi-ip:5000
"""

import os
import csv
import time
import json
import threading
import tempfile
import subprocess
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, render_template_string, jsonify, send_file, request
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

# ---------------- HARDWARE DETECTION ----------------
def detect_raspberry_pi_model():
    """Detect if running on Raspberry Pi and which model"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                if 'Pi 5' in cpuinfo or 'BCM2712' in cpuinfo:
                    return 'pi5'
                elif 'Pi 4' in cpuinfo or 'BCM2711' in cpuinfo:
                    return 'pi4'
                elif 'Pi 3' in cpuinfo or 'BCM2837' in cpuinfo:
                    return 'pi3'
                else:
                    return 'pi_other'
    except:
        pass
    return 'unknown'

PI_MODEL = detect_raspberry_pi_model()
IS_PI4 = PI_MODEL == 'pi4'
IS_PI5 = PI_MODEL == 'pi5'

# ---------------- INTERACTIVE CONFIGURATION ----------------
def get_operator_input():
    """
    Interactive prompt to get inspection parameters from operator
    """
    print("\n" + "="*60)
    print("AUTOMATED PIPELINE INSPECTION SYSTEM")
    if IS_PI4:
        print("Hardware: Raspberry Pi 4 (Optimized Mode)")
    elif IS_PI5:
        print("Hardware: Raspberry Pi 5")
    else:
        print("Hardware: Generic/Development Mode")
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
    
    # Camera mode selection for Pi 4
    camera_mode = "dual"
    if IS_PI4:
        print("\nCAMERA CONFIGURATION:")
        print("  1. Single Camera Mode (Better performance, recommended for Pi 4)")
        print("  2. Dual Camera Mode (Requires camera multiplexer or USB cameras)")
        
        while True:
            cam_choice = input(f"\nEnter choice [1/2] (default: 1): ").strip() or "1"
            if cam_choice in ["1", "2"]:
                camera_mode = "single" if cam_choice == "1" else "dual"
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
    if IS_PI4:
        print(f"  Camera Mode:      {'Single Camera' if camera_mode == 'single' else 'Dual Camera'}")
        print(f"  Hardware:         Raspberry Pi 4 (Optimized)")
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
        'camera_mode': camera_mode,
        'pipeline_length': pipeline_length,
        'velocity': velocity,
        'velocity_unit': velocity_unit,
        'velocity_mps': velocity_mps,
    }

# Get operator input
operator_config = get_operator_input()

# Apply operator configuration
MODEL_MODE = operator_config['model_mode']
CAMERA_MODE = operator_config.get('camera_mode', 'dual')
PIPELINE_LENGTH_METERS = operator_config['pipeline_length']
ROBOT_VELOCITY = operator_config['velocity']
VELOCITY_UNIT = operator_config['velocity_unit']
ROBOT_VELOCITY_MPS = operator_config['velocity_mps']
ESTIMATED_INSPECTION_DURATION_SEC = PIPELINE_LENGTH_METERS / ROBOT_VELOCITY_MPS if ROBOT_VELOCITY_MPS > 0 else 0

# Online mode: Roboflow
RF_API_URL = os.getenv("RF_API_URL", "https://detect.roboflow.com")
RF_API_KEY = os.getenv("RF_API_KEY", "")
RF_WORKSPACE = os.getenv("RF_WORKSPACE", "")
RF_WORKFLOW_ID = os.getenv("RF_WORKFLOW_ID", "")
API_KEY = RF_API_KEY

# Offline mode: Local YOLO
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/best.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")

# Initialize model based on mode
client = None
local_model = None

if MODEL_MODE == "online":
    if not ROBOFLOW_AVAILABLE:
        raise ValueError("Online mode selected but inference-sdk not installed. Install with: pip install inference-sdk")
    if not RF_API_KEY or not RF_WORKFLOW_ID:
        raise ValueError("RF_API_KEY and RF_WORKFLOW_ID must be set in .env for online mode")
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
    # Load model with CPU for Raspberry Pi 4 compatibility
    local_model = YOLO(LOCAL_MODEL_PATH)
    print(f"[INIT] On-Device AI Model initialized successfully (YOLOv11n - 68% mAP)")
    print(f"[INIT] Model path: {LOCAL_MODEL_PATH}")
    print(f"[INIT] Compute device: {YOLO_DEVICE.upper()}")
    print(f"[INIT] Robot velocity: {ROBOT_VELOCITY} {VELOCITY_UNIT} ({ROBOT_VELOCITY_MPS:.3f} m/s)")
    print(f"[INIT] Pipeline length: {PIPELINE_LENGTH_METERS}m")
    print(f"[INIT] Estimated inspection time: {ESTIMATED_INSPECTION_DURATION_SEC:.1f}s ({ESTIMATED_INSPECTION_DURATION_SEC/60:.1f} min)")
else:
    raise ValueError(f"Invalid MODEL_MODE: {MODEL_MODE}. Must be 'online' or 'offline'")

# ---------------- OPTIMIZED SETTINGS FOR PI 4 ----------------
CONF_THRESH = float(os.getenv("RF_CONF", "0.5"))
INFER_FPS = 1.0
SAVE_COOLDOWN_S = 0.5
ONLY_CLASS = ""

ENABLE_PREPROCESSING = True
ENABLE_PERSISTENCE = False
PERSISTENCE_FRAMES = 3
BLUR_THRESHOLD = 5.0
MIN_CRACK_AREA = 100

SEVERITY_CRITICAL = 0.85
SEVERITY_HIGH = 0.70
SEVERITY_MEDIUM = 0.55

BOOLEAN_DURATION_S = 1.0

# Enable/disable position tracking
ENABLE_POSITION_TRACKING = True

# Optimized resolution for Pi 4
if IS_PI4:
    CAMERA_WIDTH = 640   # Lower resolution for Pi 4
    CAMERA_HEIGHT = 480
    CAPTURE_FPS = 20     # Lower FPS for Pi 4
    FRAME_SKIP = 2       # Process every 2nd frame on Pi 4
    print(f"[INIT] Pi 4 optimizations enabled: 640x480 @ 20fps, frame skip: {FRAME_SKIP}")
else:
    CAMERA_WIDTH = 1280  # Higher resolution for Pi 5
    CAMERA_HEIGHT = 720
    CAPTURE_FPS = 30
    FRAME_SKIP = 1       # Process every frame on Pi 5

CAMERA_0_ID = int(os.getenv("CAM0_INDEX", "0"))
CAMERA_1_ID = int(os.getenv("CAM1_INDEX", "1"))

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
    Calculate estimated position of crack along pipeline using velocity-based tracking.
    
    Args:
        elapsed_sec: Time elapsed since inspection start (seconds)
        velocity_mps: Robot velocity in meters per second
        pipeline_length_m: Total pipeline length in meters
    
    Returns:
        Estimated position in meters from start (clamped to pipeline length)
    """
    position_m = velocity_mps * elapsed_sec
    # Clamp to pipeline length
    position_m = min(position_m, pipeline_length_m)
    return position_m


def draw_location_indicator(img: np.ndarray, position_m: float, total_length_m: float, 
                            timestamp: str) -> np.ndarray:
    """
    Draw location information panel on the image with progress bar.
    
    Args:
        img: Input image (BGR format)
        position_m: Current position along pipeline (meters)
        total_length_m: Total pipeline length (meters)
        timestamp: Current timestamp string
    
    Returns:
        Image with location overlay
    """
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    # Panel background
    panel_height = 80
    overlay = img_copy.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, img_copy, 0.3, 0, img_copy)
    
    # Calculate progress
    progress_pct = (position_m / total_length_m * 100) if total_length_m > 0 else 0
    progress_pct = min(100, progress_pct)
    
    # Text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    
    # Position info
    text1 = f"Position: {position_m:.2f}m / {total_length_m:.2f}m ({progress_pct:.1f}%)"
    cv2.putText(img_copy, text1, (10, 25), font, font_scale, color, thickness)
    
    # Distance remaining
    remaining = max(0, total_length_m - position_m)
    text2 = f"Remaining: {remaining:.2f}m | Time: {timestamp}"
    cv2.putText(img_copy, text2, (10, 50), font, font_scale, color, thickness)
    
    # Progress bar
    bar_x = 10
    bar_y = 60
    bar_width = w - 20
    bar_height = 15
    
    # Background bar
    cv2.rectangle(img_copy, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (80, 80, 80), -1)
    
    # Progress fill
    fill_width = int(bar_width * progress_pct / 100)
    if fill_width > 0:
        # Color gradient based on progress
        if progress_pct < 33:
            bar_color = (0, 255, 0)  # Green
        elif progress_pct < 66:
            bar_color = (0, 255, 255)  # Yellow
        else:
            bar_color = (0, 165, 255)  # Orange
        
        cv2.rectangle(img_copy, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     bar_color, -1)
    
    # Border
    cv2.rectangle(img_copy, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (255, 255, 255), 1)
    
    return img_copy


def generate_pdf_report(cracks_data: List[Dict], output_path: Path) -> bool:
    """
    Generate a professional PDF inspection report.
    
    Args:
        cracks_data: List of crack detection dictionaries
        output_path: Path where PDF will be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#283593'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
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
            ['Report Generated', time.strftime("%Y-%m-%d %H:%M:%S")],
            ['Pipeline Length', f'{PIPELINE_LENGTH_METERS:.2f} meters'],
            ['Robot Velocity', f'{ROBOT_VELOCITY:.2f} {VELOCITY_UNIT}'],
            ['Total Cracks Detected', str(len(cracks_data))],
            ['Detection Model', 'On-Device YOLOv11n (68% mAP)' if MODEL_MODE == 'offline' else 'Cloud-Based Roboflow API'],
            ['Hardware Platform', f'Raspberry Pi 4 (Optimized)' if IS_PI4 else 'Raspberry Pi 5'],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Severity Breakdown
        if cracks_data:
            story.append(Paragraph("Severity Distribution", heading_style))
            
            severity_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
            for crack in cracks_data:
                severity = crack.get('severity', 'Low')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            severity_data = [
                ['Severity Level', 'Count', 'Percentage'],
            ]
            
            for severity in ['Critical', 'High', 'Medium', 'Low']:
                count = severity_counts.get(severity, 0)
                pct = (count / len(cracks_data) * 100) if len(cracks_data) > 0 else 0
                severity_data.append([severity, str(count), f'{pct:.1f}%'])
            
            severity_table = Table(severity_data, colWidths=[2*inch, 2*inch, 2*inch])
            severity_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(severity_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Detailed Crack List
        if cracks_data:
            story.append(Paragraph("Detailed Crack List", heading_style))
            
            crack_data = [
                ['ID', 'Time', 'Position (m)', 'Confidence', 'Severity', 'Camera']
            ]
            
            for crack in cracks_data[:50]:  # Limit to first 50 to keep PDF reasonable
                crack_data.append([
                    str(crack.get('id', 'N/A')),
                    crack.get('timestamp', 'N/A'),
                    f"{crack.get('position_m', 0):.2f}",
                    f"{crack.get('confidence', 0):.2f}",
                    crack.get('severity', 'N/A'),
                    crack.get('camera', 'N/A')
                ])
            
            if len(cracks_data) > 50:
                crack_data.append(['...', f'{len(cracks_data) - 50} more cracks not shown', '', '', '', ''])
            
            crack_table = Table(crack_data, colWidths=[0.5*inch, 1.3*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
            crack_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.lightgrey])
            ]))
            story.append(crack_table)
        else:
            story.append(Paragraph("No cracks detected during inspection.", normal_style))
        
        # Build PDF
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to generate PDF: {e}")
        return False


def parse_severity(conf: float) -> str:
    if conf >= SEVERITY_CRITICAL:
        return "Critical"
    elif conf >= SEVERITY_HIGH:
        return "High"
    elif conf >= SEVERITY_MEDIUM:
        return "Medium"
    else:
        return "Low"


def compute_laplacian_variance(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if not ENABLE_PREPROCESSING:
        return frame
    
    # Downscale for Pi 4 performance
    if IS_PI4 and frame.shape[0] > 480:
        scale = 480 / frame.shape[0]
        new_w = int(frame.shape[1] * scale)
        frame = cv2.resize(frame, (new_w, 480))
    
    # Denoise
    frame = cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10, 
                                            templateWindowSize=7, searchWindowSize=21)
    
    # Enhance contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Slight sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """
    Extract prediction data from either Roboflow or YOLO result.
    Returns list of dicts with keys: class_name, confidence, x, y, width, height
    """
    predictions = []
    
    # Check if it's a YOLO Results object
    if hasattr(result, 'boxes'):
        # Ultralytics YOLO format
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get box coordinates
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Convert to center format
                x1, y1, x2, y2 = xyxy
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Get class name from model
                class_name = result.names[cls] if hasattr(result, 'names') else f"class_{cls}"
                
                predictions.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'x': float(x_center),
                    'y': float(y_center),
                    'width': float(width),
                    'height': float(height)
                })
    
    # Check if it's a Roboflow workflow result
    elif isinstance(result, dict):
        # Roboflow format
        if 'output' in result and 'predictions' in result['output']:
            for pred in result['output']['predictions']:
                predictions.append({
                    'class_name': pred.get('class', 'crack'),
                    'confidence': pred.get('confidence', 0.0),
                    'x': pred.get('x', 0),
                    'y': pred.get('y', 0),
                    'width': pred.get('width', 0),
                    'height': pred.get('height', 0)
                })
    
    return predictions


# ---------------- GLOBAL STATE ----------------
crack_log = []
csv_path = OUT_BASE / f"inspection_log_{stamp()}.csv"

csv_headers = [
    "id", "timestamp", "camera", "confidence", "severity",
    "class_name", "x", "y", "width", "height",
    "elapsed_sec", "position_m", "progress_pct"
]

with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(csv_headers)

crack_lock = threading.Lock()
next_id = 1
last_cam0_detect_t = 0.0
last_cam1_detect_t = 0.0

detection_flags = {"cam0": False, "cam1": False}
flag_timers = {"cam0": 0.0, "cam1": 0.0}

# Inspection timing
inspection_start_time = time.time()

def set_flag(cam: str):
    detection_flags[cam] = True
    flag_timers[cam] = time.time()


def update_flags():
    now = time.time()
    for cam in ["cam0", "cam1"]:
        if detection_flags[cam] and (now - flag_timers[cam] > BOOLEAN_DURATION_S):
            detection_flags[cam] = False


def log_detection(
    camera: str,
    conf: float,
    class_name: str,
    x: float, y: float, w: float, h: float,
    img: Optional[np.ndarray] = None
):
    global next_id, last_cam0_detect_t, last_cam1_detect_t
    
    now_t = time.time()
    elapsed_sec = now_t - inspection_start_time
    
    # Calculate position if tracking enabled
    if ENABLE_POSITION_TRACKING:
        position_m = estimate_crack_position(elapsed_sec, ROBOT_VELOCITY_MPS, PIPELINE_LENGTH_METERS)
        progress_pct = (position_m / PIPELINE_LENGTH_METERS * 100) if PIPELINE_LENGTH_METERS > 0 else 0
    else:
        position_m = 0.0
        progress_pct = 0.0
    
    # Apply cooldown
    if camera == "CAM0":
        if now_t - last_cam0_detect_t < SAVE_COOLDOWN_S:
            return
        last_cam0_detect_t = now_t
    else:
        if now_t - last_cam1_detect_t < SAVE_COOLDOWN_S:
            return
        last_cam1_detect_t = now_t
    
    sev = parse_severity(conf)
    ts_str = time.strftime("%H:%M:%S")
    
    with crack_lock:
        det_id = next_id
        next_id += 1
        
        det_record = {
            "id": det_id,
            "timestamp": ts_str,
            "camera": camera,
            "confidence": conf,
            "severity": sev,
            "class_name": class_name,
            "x": x, "y": y,
            "width": w, "height": h,
            "elapsed_sec": elapsed_sec,
            "position_m": position_m,
            "progress_pct": progress_pct
        }
        crack_log.append(det_record)
        
        # Save image with location indicator
        if img is not None:
            img_with_location = draw_location_indicator(img, position_m, PIPELINE_LENGTH_METERS, ts_str)
            
            if camera == "CAM0":
                img_path = FOUND_DIR_CAM0 / f"crack_{det_id:04d}_{ts_str.replace(':', '')}.jpg"
            else:
                img_path = FOUND_DIR_CAM1 / f"crack_{det_id:04d}_{ts_str.replace(':', '')}.jpg"
            
            cv2.imwrite(str(img_path), img_with_location)
            det_record["image_path"] = str(img_path)
        
        # CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                det_id, ts_str, camera, f"{conf:.3f}", sev,
                class_name, f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}",
                f"{elapsed_sec:.2f}", f"{position_m:.2f}", f"{progress_pct:.1f}"
            ])
    
    set_flag("cam0" if camera == "CAM0" else "cam1")


# ---------------- CAMERA THREADS ----------------
stop_event = threading.Event()
cam0_frame = None
cam1_frame = None
cam0_lock = threading.Lock()
cam1_lock = threading.Lock()

frame_counter_cam0 = 0
frame_counter_cam1 = 0


def cam_loop(camera_id: int, camera_name: str, lock, global_frame_var: str):
    """
    Camera capture loop with frame skipping optimization for Pi 4.
    """
    global cam0_frame, cam1_frame, frame_counter_cam0, frame_counter_cam1
    
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    
    if not cap.isOpened():
        print(f"[{camera_name}] Failed to open camera {camera_id}")
        return
    
    print(f"[{camera_name}] Camera {camera_id} opened successfully")
    
    infer_timer = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Frame skipping for performance
        if camera_name == "CAM0":
            frame_counter_cam0 += 1
            if frame_counter_cam0 % FRAME_SKIP != 0:
                continue
        else:
            frame_counter_cam1 += 1
            if frame_counter_cam1 % FRAME_SKIP != 0:
                continue
        
        # Update global frame for streaming
        with lock:
            if global_frame_var == 'cam0':
                cam0_frame = frame.copy()
            else:
                cam1_frame = frame.copy()
        
        # Inference throttle
        if time.time() - infer_timer < (1.0 / INFER_FPS):
            continue
        infer_timer = time.time()
        
        # Preprocessing
        processed = preprocess_frame(frame)
        
        # Blur check
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        blur_score = compute_laplacian_variance(gray)
        if blur_score < BLUR_THRESHOLD:
            continue
        
        # Model inference
        try:
            if MODEL_MODE == "online" and client:
                # Online inference (Roboflow)
                _, buffer = cv2.imencode('.jpg', processed)
                result = client.run_workflow(
                    workspace_name=RF_WORKSPACE,
                    workflow_id=RF_WORKFLOW_ID,
                    images={"image": buffer.tobytes()}
                )
                preds = extract_predictions(result)
            
            elif MODEL_MODE == "offline" and local_model:
                # Offline inference (YOLO)
                results = local_model(processed, conf=CONF_THRESH, device=YOLO_DEVICE, verbose=False)
                if len(results) > 0:
                    preds = extract_predictions(results[0])
                else:
                    preds = []
            else:
                preds = []
            
            # Process detections
            for pred in preds:
                conf = pred['confidence']
                class_name = pred['class_name']
                
                if ONLY_CLASS and class_name != ONLY_CLASS:
                    continue
                if conf < CONF_THRESH:
                    continue
                
                x, y = pred['x'], pred['y']
                w, h = pred['width'], pred['height']
                
                # Area filter
                area = w * h
                if area < MIN_CRACK_AREA:
                    continue
                
                # Log detection
                log_detection(camera_name, conf, class_name, x, y, w, h, frame)
        
        except Exception as e:
            print(f"[{camera_name}] Inference error: {e}")
            continue
    
    cap.release()
    print(f"[{camera_name}] Camera {camera_id} released")


# ---------------- FLASK APP ----------------
app = Flask(__name__)

def generate_frames(camera_index: int):
    """Generate frames for MJPEG streaming"""
    while True:
        if camera_index == 0:
            with cam0_lock:
                frame = cam0_frame
        else:
            with cam1_lock:
                frame = cam1_frame
        
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Add detection overlay
        display_frame = frame.copy()
        
        # Add detection indicator
        if camera_index == 0 and detection_flags["cam0"]:
            cv2.putText(display_frame, "CRACK DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif camera_index == 1 and detection_flags["cam1"]:
            cv2.putText(display_frame, "CRACK DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Main dashboard page"""
    model_display = "On-Device YOLOv11n (68% mAP)" if MODEL_MODE == "offline" else "Cloud-Based Roboflow API"
    hardware_display = "Raspberry Pi 4 (Optimized)" if IS_PI4 else "Raspberry Pi 5"
    
    # Single or dual camera display
    if CAMERA_MODE == "single":
        video_section = '''
        <div class="video-container-single">
            <div class="video-wrapper">
                <h3>Camera Feed</h3>
                <img src="/video_feed/0" alt="Camera 0">
            </div>
        </div>
        '''
    else:
        video_section = '''
        <div class="video-container">
            <div class="video-wrapper">
                <h3>Camera 0</h3>
                <img src="/video_feed/0" alt="Camera 0">
            </div>
            <div class="video-wrapper">
                <h3>Camera 1</h3>
                <img src="/video_feed/1" alt="Camera 1">
            </div>
        </div>
        '''
    
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Automated Pipeline Inspection System</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .header {{
            background: white;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{ 
            color: #667eea;
            font-size: 28px;
        }}
        .model-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }}
        .model-badge.offline {{
            background: #4caf50;
            color: white;
        }}
        .model-badge.online {{
            background: #2196f3;
            color: white;
        }}
        .hardware-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            background: #ff9800;
            color: white;
            margin-left: 10px;
        }}
        .stats-bar {{
            background: white;
            padding: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .video-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }}
        .video-container-single {{
            display: flex;
            justify-content: center;
            padding: 20px;
        }}
        .video-wrapper {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .video-wrapper h3 {{
            margin-bottom: 10px;
            color: #667eea;
        }}
        .video-wrapper img {{
            width: 100%;
            border-radius: 8px;
        }}
        .crack-list {{
            background: white;
            margin: 10px;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }}
        .crack-item {{
            padding: 10px;
            margin: 5px 0;
            background: #f5f5f5;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .crack-item.critical {{ border-left-color: #f44336; }}
        .crack-item.high {{ border-left-color: #ff9800; }}
        .crack-item.medium {{ border-left-color: #ffc107; }}
        .crack-item.low {{ border-left-color: #4caf50; }}
        .report-button {{
            display: block;
            margin: 20px auto;
            padding: 15px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s;
        }}
        .report-button:hover {{
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(118, 75, 162, 0.4);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>
            Automated Pipeline Inspection System
            <span class="model-badge {MODEL_MODE}">{model_display}</span>
            <span class="hardware-badge">{hardware_display}</span>
        </h1>
    </div>
    
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-label">Pipeline Length</div>
            <div class="stat-value" id="pipeline-length">{PIPELINE_LENGTH_METERS:.1f}m</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Current Position</div>
            <div class="stat-value" id="current-position">0.0m</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Progress</div>
            <div class="stat-value" id="progress">0%</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Total Cracks</div>
            <div class="stat-value" id="total-cracks">0</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Critical</div>
            <div class="stat-value" id="critical-cracks" style="color: #f44336;">0</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">High</div>
            <div class="stat-value" id="high-cracks" style="color: #ff9800;">0</div>
        </div>
    </div>
    
    {video_section}
    
    <button class="report-button" onclick="generateReport()">Generate PDF Report</button>
    
    <div class="crack-list">
        <h2>Recent Detections</h2>
        <div id="crack-items"></div>
    </div>
    
    <script>
        function updateStats() {{
            fetch('/api/cracks')
                .then(response => response.json())
                .then(data => {{
                    // Calculate elapsed time
                    const elapsed = (Date.now() / 1000) - {inspection_start_time};
                    const position = {ROBOT_VELOCITY_MPS} * elapsed;
                    const clampedPosition = Math.min(position, {PIPELINE_LENGTH_METERS});
                    const progress = (clampedPosition / {PIPELINE_LENGTH_METERS} * 100).toFixed(1);
                    
                    document.getElementById('current-position').textContent = clampedPosition.toFixed(1) + 'm';
                    document.getElementById('progress').textContent = progress + '%';
                    document.getElementById('total-cracks').textContent = data.length;
                    
                    // Count by severity
                    let critical = 0, high = 0;
                    data.forEach(crack => {{
                        if (crack.severity === 'Critical') critical++;
                        else if (crack.severity === 'High') high++;
                    }});
                    
                    document.getElementById('critical-cracks').textContent = critical;
                    document.getElementById('high-cracks').textContent = high;
                    
                    // Update crack list (last 10)
                    const crackItems = document.getElementById('crack-items');
                    crackItems.innerHTML = '';
                    data.slice(-10).reverse().forEach(crack => {{
                        const div = document.createElement('div');
                        div.className = 'crack-item ' + crack.severity.toLowerCase();
                        div.innerHTML = `
                            <strong>ID #${{crack.id}}</strong> - ${{crack.timestamp}} - 
                            ${{crack.camera}} - Position: ${{crack.position_m}}m - 
                            Confidence: ${{(crack.confidence * 100).toFixed(1)}}% - 
                            <span style="color: ${{getSeverityColor(crack.severity)}}">${{crack.severity}}</span>
                        `;
                        crackItems.appendChild(div);
                    }});
                }});
        }}
        
        function getSeverityColor(severity) {{
            const colors = {{
                'Critical': '#f44336',
                'High': '#ff9800',
                'Medium': '#ffc107',
                'Low': '#4caf50'
            }};
            return colors[severity] || '#666';
        }}
        
        function generateReport() {{
            window.location.href = '/generate_report';
        }}
        
        // Update every 2 seconds
        updateStats();
        setInterval(updateStats, 2000);
    </script>
</body>
</html>
    '''
    return html


@app.route('/video_feed/<int:camera>')
def video_feed(camera):
    """Video streaming route"""
    return Response(generate_frames(camera),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cracks')
def api_cracks():
    """Return crack detection data as JSON"""
    with crack_lock:
        data = crack_log.copy()
    return jsonify(data)


@app.route('/generate_report')
def generate_report_route():
    """Generate and download PDF report"""
    try:
        with crack_lock:
            cracks_data = crack_log.copy()
        
        report_path = REPORTS_DIR / f"inspection_report_{stamp()}.pdf"
        success = generate_pdf_report(cracks_data, report_path)
        
        if success and report_path.exists():
            return send_file(
                report_path,
                as_attachment=True,
                download_name=f"inspection_report_{stamp()}.pdf",
                mimetype='application/pdf'
            )
        else:
            return "Failed to generate report", 500
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        return f"Error: {e}", 500


@app.route('/crack_image/<int:crack_id>')
def crack_image(crack_id):
    """Serve crack image by ID"""
    with crack_lock:
        crack = next((c for c in crack_log if c['id'] == crack_id), None)
    
    if crack and 'image_path' in crack:
        img_path = Path(crack['image_path'])
        if img_path.exists():
            return send_file(img_path, mimetype='image/jpeg')
    
    return "Image not found", 404


# ---------------- MAIN ----------------
def main():
    print(f"\n{'='*60}")
    print(f"AUTOMATED PIPELINE INSPECTION SYSTEM - {'Pi 4 Optimized' if IS_PI4 else 'Pi 5'}")
    print(f"{'='*60}")
    print(f"Model: {MODEL_MODE.upper()}")
    print(f"Camera Mode: {CAMERA_MODE.upper()}")
    print(f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAPTURE_FPS}fps")
    if IS_PI4:
        print(f"Frame Skip: Processing every {FRAME_SKIP} frame(s)")
    print(f"Pipeline: {PIPELINE_LENGTH_METERS}m")
    print(f"Velocity: {ROBOT_VELOCITY_MPS:.3f} m/s")
    print(f"Web Interface: http://0.0.0.0:{FLASK_PORT}")
    print(f"{'='*60}\n")
    
    # Start camera threads
    cam0_thread = threading.Thread(target=cam_loop, args=(CAMERA_0_ID, "CAM0", cam0_lock, 'cam0'), daemon=True)
    cam0_thread.start()
    
    if CAMERA_MODE == "dual":
        cam1_thread = threading.Thread(target=cam_loop, args=(CAMERA_1_ID, "CAM1", cam1_lock, 'cam1'), daemon=True)
        cam1_thread.start()
    
    # Flag update thread
    def flag_updater():
        while not stop_event.is_set():
            update_flags()
            time.sleep(0.1)
    
    flag_thread = threading.Thread(target=flag_updater, daemon=True)
    flag_thread.start()
    
    # Start Flask
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping system...")
        stop_event.set()
        print("[SHUTDOWN] System stopped")


if __name__ == "__main__":
    main()
