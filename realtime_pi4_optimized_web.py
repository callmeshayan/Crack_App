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

# Picamera2 for Pi CSI camera (libcamera / imx708 / Module 3)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Will try cv2.VideoCapture fallback.")

# Conditional imports for online/offline models
# inference_sdk does not support Python 3.13 — use direct HTTP instead
import base64 as _b64mod
import requests as _requests

ROBOFLOW_AVAILABLE = True  # always available via direct HTTP

class InferenceHTTPClient:
    """Minimal Roboflow workflow client using plain requests (no inference_sdk)."""
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key

    def run_workflow(self, workspace_name: str, workflow_id: str, images: dict) -> list:
        url = f"{self.api_url}/infer/workflows/{workspace_name}/{workflow_id}"
        # Build image inputs
        inputs = {}
        for key, value in images.items():
            if isinstance(value, bytes):
                b64 = _b64mod.b64encode(value).decode('utf-8')
            else:
                b64 = value
            inputs[key] = {"type": "base64", "value": b64}
        payload = {"inputs": inputs, "api_key": self.api_key}
        resp = _requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Return outputs list (same shape as inference_sdk)
        return data.get("outputs", [data])

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
    
    # Pi 4 has only one on-board CSI camera connector — always single camera
    camera_mode = "single"
    
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
        print(f"  Camera Mode:      Single Camera (on-board CSI)")
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
# Configuration from environment — operator configures via web UI at runtime
MODEL_MODE = os.getenv("MODEL_MODE", "offline").strip().lower()
CAMERA_MODE = "single"  # Pi 4 has one on-board CSI connector
_default_velocity = float(os.getenv("ROBOT_VELOCITY", "0.167"))
VELOCITY_UNIT = os.getenv("VELOCITY_UNIT", "m/s").lower()
ROBOT_VELOCITY = _default_velocity
if VELOCITY_UNIT == "km/h":
    ROBOT_VELOCITY_MPS = _default_velocity / 3.6
else:
    ROBOT_VELOCITY_MPS = _default_velocity
PIPELINE_LENGTH_METERS = float(os.getenv("PIPELINE_LENGTH", "100.0"))
ESTIMATED_INSPECTION_DURATION_SEC = (
    PIPELINE_LENGTH_METERS / ROBOT_VELOCITY_MPS if ROBOT_VELOCITY_MPS > 0 else 600.0
)

# Online mode: Roboflow
RF_API_URL = os.getenv("RF_API_URL", "https://detect.roboflow.com")
RF_API_KEY = os.getenv("RF_API_KEY", "")
RF_WORKSPACE = os.getenv("RF_WORKSPACE", "")
RF_WORKFLOW_ID = os.getenv("RF_WORKFLOW_ID", "")
API_KEY = RF_API_KEY

# Offline mode: Local YOLO
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/best.pt")
PVC_MODEL_PATH = os.getenv("PVC_MODEL_PATH", "models/yolo11n_pvc_trained_best.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")

# Model storage for dynamic switching
available_models = {}
current_model_type = "metal"  # default

# Initialize model based on mode (lenient — crashes handled at runtime)
client = None
local_model = None

if MODEL_MODE == "online":
    if not ROBOFLOW_AVAILABLE:
        print("WARNING: inference-sdk not installed; online mode unavailable.")
    elif not RF_API_KEY:
        print("WARNING: RF_API_KEY not set; online mode will fail until configured.")
    else:
        try:
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=API_KEY,
            )
            print("[INIT] Cloud-Based AI Model initialized successfully")
        except Exception as _e:
            print(f"WARNING: Could not init Roboflow client: {_e}")
elif MODEL_MODE == "offline":
    if not ULTRALYTICS_AVAILABLE:
        print("WARNING: ultralytics not installed; offline model unavailable.")
    elif not Path(LOCAL_MODEL_PATH).exists():
        print(f"WARNING: Offline model not found at {LOCAL_MODEL_PATH}; will retry on start.")
    else:
        try:
            local_model = YOLO(LOCAL_MODEL_PATH)
            available_models['metal'] = local_model
            print(f"[INIT] On-Device AI Model initialized (YOLOv11n - 68% mAP)")
            
            # Load PVC model if available
            if Path(PVC_MODEL_PATH).exists():
                pvc_model = YOLO(PVC_MODEL_PATH)
                available_models['pvc'] = pvc_model
                print(f"[INIT] PVC Trained Model also loaded (35% mAP)")
        except Exception as _e:
            print(f"WARNING: Could not load offline model: {_e}")
else:
    print(f"WARNING: Unknown MODEL_MODE '{MODEL_MODE}'; defaulting to offline.")
    MODEL_MODE = "offline"

# ---------------- OPTIMIZED SETTINGS FOR PI 4 ----------------
CONF_THRESH = float(os.getenv("RF_CONF", "0.25"))
INFER_FPS = 3.0
SAVE_COOLDOWN_S = 0.5
ONLY_CLASS = ""

ENABLE_PREPROCESSING = False  # Heavy denoising masks crack features and slows Pi 4
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
    FRAME_SKIP = 1       # Process every frame on Pi 4
    print(f"[INIT] Pi 4 optimizations enabled: 640x480 @ 20fps, frame skip: {FRAME_SKIP}")
else:
    CAMERA_WIDTH = 1280  # Higher resolution for Pi 5
    CAMERA_HEIGHT = 720
    CAPTURE_FPS = 30
    FRAME_SKIP = 1       # Process every frame on Pi 5

CAMERA_0_ID = int(os.getenv("CAM0_INDEX", "0"))

DASHBOARD_INTERVAL_S = 5.0

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

OUT_BASE = Path("data/realtime_results")
FOUND_DIR_CAM0 = OUT_BASE / "camera0_found"
REALTIME_FOUND_DIR_CAM0 = OUT_BASE / "camera0_realtime"
REPORTS_DIR = OUT_BASE / "reports"

for p in [FOUND_DIR_CAM0, REALTIME_FOUND_DIR_CAM0, REPORTS_DIR]:
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
    Extract prediction data from either Roboflow workflow or YOLO result.
    Returns list of dicts with keys: class_name, confidence, x, y, width, height
    """
    # YOLO Results object
    if hasattr(result, 'boxes'):
        predictions = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy
                class_name = result.names[cls] if hasattr(result, 'names') else f"class_{cls}"
                predictions.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'x': float((x1 + x2) / 2),
                    'y': float((y1 + y2) / 2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                })
        return predictions

    # Roboflow workflow returns a list of output dicts
    if isinstance(result, list):
        for item in result:
            preds = extract_predictions(item)
            if preds:
                return preds
        return []

    if isinstance(result, dict):
        # Look for a 'predictions' key at any level
        raw = result.get('predictions')
        if isinstance(raw, list) and raw:
            out = []
            for p in raw:
                out.append({
                    'class_name': p.get('class', p.get('class_name', 'crack')),
                    'confidence': float(p.get('confidence', 0.0)),
                    'x': float(p.get('x', 0)),
                    'y': float(p.get('y', 0)),
                    'width': float(p.get('width', 0)),
                    'height': float(p.get('height', 0))
                })
            return out
        # Recurse into nested dicts
        for v in result.values():
            if isinstance(v, (dict, list)):
                preds = extract_predictions(v)
                if preds:
                    return preds

    return []


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

detection_flags = {"cam0": False}
flag_timers = {"cam0": 0.0}
cam0_boxes = []          # latest bounding boxes: list of (x1, y1, x2, y2, conf)
cam0_boxes_lock = threading.Lock()
cam0_boxes_expire = 0.0  # time after which boxes should be cleared

# Inspection timing
inspection_start_time = time.time()
_app_start_time = time.time()  # used to show friendly "initializing" message on startup

def set_flag(cam: str):
    detection_flags[cam] = True
    flag_timers[cam] = time.time()


def update_flags():
    now = time.time()
    if detection_flags["cam0"] and (now - flag_timers["cam0"] > BOOLEAN_DURATION_S):
        detection_flags["cam0"] = False


def log_detection(
    camera: str,
    conf: float,
    class_name: str,
    x: float, y: float, w: float, h: float,
    img: Optional[np.ndarray] = None
):
    global next_id, last_cam0_detect_t
    
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
    if now_t - last_cam0_detect_t < SAVE_COOLDOWN_S:
        return
    last_cam0_detect_t = now_t
    
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
            img_path = FOUND_DIR_CAM0 / f"crack_{det_id:04d}_{ts_str.replace(':', '')}.jpg"
            cv2.imwrite(str(img_path), img_with_location)
            det_record["image_path"] = str(img_path)
        
        # CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                det_id, ts_str, camera, f"{conf:.3f}", sev,
                class_name, f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}",
                f"{elapsed_sec:.2f}", f"{position_m:.2f}", f"{progress_pct:.1f}"
            ])
    
    set_flag("cam0")


# ---------------- CAMERA THREADS ----------------
stop_event = threading.Event()
cam_stop_event = threading.Event()   # stops camera capture (only on full app shutdown)
inspection_paused = threading.Event()  # set = paused
cam0_frame = None
cam0_lock = threading.Lock()
_cam0_thread_started = False

frame_counter_cam0 = 0


def _run_inference_and_log(frame, camera_name):
    """Run inference on a frame and log any detections."""
    processed = preprocess_frame(frame)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if compute_laplacian_variance(gray) < BLUR_THRESHOLD:
        return
    try:
        if MODEL_MODE == "online" and client:
            _, buffer = cv2.imencode('.jpg', processed)
            result = client.run_workflow(
                workspace_name=RF_WORKSPACE,
                workflow_id=RF_WORKFLOW_ID,
                images={"image": buffer.tobytes()}
            )
            preds = extract_predictions(result)
            # Normalise field name: Roboflow uses 'class', code uses 'class_name'
            for p in preds:
                if 'class' in p and 'class_name' not in p:
                    p['class_name'] = p['class']
        elif MODEL_MODE == "offline" and local_model:
            # Use currently selected model
            active_model = available_models.get(current_model_type, local_model)
            results = active_model(processed, conf=CONF_THRESH, device=YOLO_DEVICE, verbose=False)
            preds = extract_predictions(results[0]) if len(results) > 0 else []
        else:
            preds = []
        global cam0_boxes, cam0_boxes_expire
        new_boxes = []
        for pred in preds:
            conf = pred['confidence']
            class_name = pred['class_name']
            if ONLY_CLASS and class_name != ONLY_CLASS:
                continue
            if conf < CONF_THRESH:
                continue
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            if w * h < MIN_CRACK_AREA:
                continue
            # Store box for live overlay (centre-based → corner coords)
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            new_boxes.append((x1, y1, x2, y2, conf))
            log_detection(camera_name, conf, class_name, x, y, w, h, frame)
        if new_boxes:
            with cam0_boxes_lock:
                cam0_boxes = new_boxes
                cam0_boxes_expire = time.time() + 1.5  # show box for 1.5 s
    except Exception as e:
        print(f"[{camera_name}] Inference error: {e}")


def _ensure_camera_available():
    """
    Pre-flight: kill any stale Python processes holding the CSI camera, and
    if WirePlumber is holding libcamera, restart it so it picks up our
    config override that disables libcamera monitoring.
    """
    import signal as _signal
    my_pid = os.getpid()
    camera_devs = ['/dev/video0', '/dev/video1', '/dev/media4']

    # Step 1: kill stale Python processes that have the camera open
    for dev in camera_devs:
        try:
            result = subprocess.run(['fuser', dev], capture_output=True, text=True, timeout=3)
            for pid_str in result.stdout.split():
                try:
                    pid = int(pid_str)
                    if pid == my_pid:
                        continue
                    proc = subprocess.run(
                        ['ps', '-p', str(pid), '-o', 'comm='],
                        capture_output=True, text=True, timeout=3
                    )
                    if 'python' in proc.stdout.lower():
                        print(f"[CAMERA] Killing stale Python process {pid} holding {dev}", flush=True)
                        os.kill(pid, _signal.SIGTERM)
                        time.sleep(0.5)
                except (ValueError, OSError):
                    pass
        except Exception:
            pass

    # Step 2: if WirePlumber still holds the camera, restart it so our
    # ~/.config/wireplumber/wireplumber.conf.d/51-disable-libcamera.conf takes effect
    time.sleep(0.3)
    wireplumber_restarted = False
    for dev in camera_devs:
        if wireplumber_restarted:
            break
        try:
            result = subprocess.run(['fuser', dev], capture_output=True, text=True, timeout=3)
            for pid_str in result.stdout.split():
                try:
                    pid = int(pid_str)
                    if pid == my_pid:
                        continue
                    proc = subprocess.run(
                        ['ps', '-p', str(pid), '-o', 'comm='],
                        capture_output=True, text=True, timeout=3
                    )
                    if 'wireplumber' in proc.stdout.lower():
                        print("[CAMERA] WirePlumber is holding the camera — restarting it "
                              "to apply libcamera-disable config...", flush=True)
                        subprocess.run(
                            ['systemctl', '--user', 'restart', 'wireplumber'],
                            timeout=15
                        )
                        time.sleep(2)
                        print("[CAMERA] WirePlumber restarted. Camera should now be free.", flush=True)
                        wireplumber_restarted = True
                        break
                except (ValueError, OSError):
                    pass
        except Exception:
            pass


def cam_loop(camera_id: int, camera_name: str):
    """
    Camera capture loop — always captures frames for live feed.
    Uses picamera2 for Pi CSI cameras, falls back to cv2.VideoCapture.
    Inference only runs when the inspection project is active and not paused.
    """
    global cam0_frame, frame_counter_cam0

    picam = None
    use_picamera2 = False

    if PICAMERA2_AVAILABLE:
        # Release any competing processes (stale Python runs, WirePlumber) before
        # attempting to claim the camera through libcamera.
        _ensure_camera_available()
        # ---- picamera2 path (Pi Camera Module 3 / CSI) ----
        print(f"[{camera_name}] Opening via picamera2 (libcamera)", flush=True)
        for attempt in range(5):
            try:
                if picam is not None:
                    try:
                        picam.close()
                    except Exception:
                        pass
                    picam = None
                if attempt > 0:
                    print(f"[{camera_name}] Retry {attempt}/4 after 2s...", flush=True)
                    time.sleep(2)
                picam = Picamera2(camera_id)
                cfg = picam.create_preview_configuration(
                    main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
                )
                picam.configure(cfg)
                picam.start()
                use_picamera2 = True
                print(f"[{camera_name}] picamera2 started successfully", flush=True)
                break
            except Exception as e:
                print(f"[{camera_name}] picamera2 attempt {attempt+1} failed: {e}", flush=True)
                if picam is not None:
                    try:
                        picam.close()
                    except Exception:
                        pass
                    picam = None
        if not use_picamera2:
            print(f"[{camera_name}] picamera2 could not start after 5 attempts; falling back to cv2.VideoCapture.", flush=True)

    if use_picamera2 and picam is not None:
        infer_timer = time.time()
        try:
            while not cam_stop_event.is_set():
                try:
                    rgb = picam.capture_array()
                except Exception as e:
                    print(f"[{camera_name}] Capture error: {e}")
                    time.sleep(0.1)
                    continue

                # picamera2 gives RGB — convert to BGR for OpenCV
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                with cam0_lock:
                    cam0_frame = frame.copy()

                # Only run inference when project is active and not paused
                with config_lock:
                    _started = project_config.get("started", False)
                if not _started or inspection_paused.is_set():
                    continue

                if time.time() - infer_timer >= (1.0 / INFER_FPS):
                    infer_timer = time.time()
                    _run_inference_and_log(frame, camera_name)
        finally:
            picam.stop()
            print(f"[{camera_name}] picamera2 stopped")
        return

    # ---- cv2.VideoCapture fallback (USB webcam / no CSI camera) ----
    cap = None
    for cv2_attempt in range(5):
        if cv2_attempt > 0:
            print(f"[{camera_name}] cv2 retry {cv2_attempt}/4 after 2s...", flush=True)
            time.sleep(2)
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        if cap.isOpened():
            break
        cap.release()
        cap = None
    if cap is None or not cap.isOpened():
        print(f"[{camera_name}] Failed to open camera {camera_id} after 5 attempts", flush=True)
        return
    print(f"[{camera_name}] cv2.VideoCapture opened camera {camera_id}", flush=True)
    infer_timer = time.time()
    try:
        while not cam_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with cam0_lock:
                cam0_frame = frame.copy()

            # Only run inference when project is active and not paused
            with config_lock:
                _started = project_config.get("started", False)
            if not _started or inspection_paused.is_set():
                continue

            if time.time() - infer_timer >= (1.0 / INFER_FPS):
                infer_timer = time.time()
                _run_inference_and_log(frame, camera_name)
    finally:
        cap.release()
        print(f"[{camera_name}] Camera {camera_id} released")


# ---------------- FLASK WEB SERVER ----------------
app = Flask(__name__)

# Project configuration state — set via web UI
project_config = {
    "initialized": False,
    "pipeline_length": PIPELINE_LENGTH_METERS,
    "robot_velocity": ROBOT_VELOCITY_MPS,
    "inspection_duration": ESTIMATED_INSPECTION_DURATION_SEC,
    "model_id": MODEL_MODE,
    "started": False,
    "paused": False,
}
config_lock = threading.Lock()

CONFIG_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Project Configuration - Pipeline Crack Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .config-container {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 2px solid rgba(0,255,136,0.3);
            box-shadow: 0 10px 50px rgba(0,0,0,0.5);
        }
        h1 { color: #00ff88; font-size: 2.2em; text-align: center; margin-bottom: 10px;
             text-shadow: 0 0 20px rgba(0,255,136,0.5); }
        .subtitle { text-align: center; opacity: 0.7; margin-bottom: 30px; }
        .form-group { margin-bottom: 25px; }
        label { display: block; margin-bottom: 8px; color: #00ff88; font-weight: 600; }
        .help-text { font-size: 0.85em; opacity: 0.7; margin-top: 5px; }
        input[type="number"], select {
            width: 100%; padding: 12px; border-radius: 8px;
            border: 2px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05); color: #fff; font-size: 1em;
            transition: all 0.3s;
        }
        input[type="number"]:focus, select:focus {
            outline: none; border-color: #00ff88;
            background: rgba(0,255,136,0.05);
        }
        select option { background: #1a1a2e; }
        .btn {
            width: 100%; padding: 15px; border-radius: 8px; border: none;
            background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
            color: #000; font-size: 1.2em; font-weight: bold;
            cursor: pointer; transition: all 0.3s; margin-top: 20px;
        }
        .btn:hover { transform: translateY(-2px);
                     box-shadow: 0 5px 20px rgba(0,255,136,0.4); }
        .info-box {
            background: rgba(0,136,255,0.1); border-left: 4px solid #0088ff;
            padding: 15px; border-radius: 8px; margin-bottom: 25px;
        }
        .info-box h3 { color: #0088ff; margin-bottom: 8px; }
        .duration-box {
            background: rgba(0,255,136,0.1); padding: 15px;
            border-radius: 8px; margin: 20px 0;
        }
        .error { background: rgba(255,0,0,0.1); border: 1px solid #f00;
                 padding: 10px; border-radius: 5px; margin-top: 10px; display: none; }
    </style>
</head>
<body>
    <div class="config-container">
        <h1>🔧 Project Setup</h1>
        <p class="subtitle">Raspberry Pi 4 — Single Camera Inspection</p>

        <div class="info-box">
            <h3>ℹ️ Before You Start</h3>
            <p>Configure pipeline parameters for accurate crack position tracking. Settings take effect immediately on start.</p>
        </div>

        <form id="configForm">
            <div class="form-group">
                <label for="pipelineLength">Pipeline Length (meters)</label>
                <input type="number" id="pipelineLength" value="100" min="1" max="10000" step="0.1" required>
                <p class="help-text">Total length of the pipeline to inspect</p>
            </div>

            <div class="form-group">
                <label for="robotVelocity">Robot Velocity</label>
                <div style="display:flex;gap:10px;">
                    <input type="number" id="robotVelocity" value="0.6" min="0.001" max="100" step="0.001" required style="flex:1;">
                    <select id="velocityUnit" style="width:100px;">
                        <option value="kmh" selected>km/h</option>
                        <option value="ms">m/s</option>
                    </select>
                </div>
                <p class="help-text">Speed of the robot (Default: 0.6 km/h = 0.167 m/s)</p>
            </div>

            <div class="form-group">
                <label for="modelSelect">Detection Model</label>
                <select id="modelSelect" required>
                    <option value="offline" selected>pipe_crack_ai — On-Device (68% mAP)</option>
                    <option value="online">Roboflow Cloud</option>
                </select>
                <p class="help-text">Select the detection model</p>
            </div>

            <div class="duration-box">
                <strong>Estimated Inspection Duration:</strong>
                <span id="durationValue">600</span> s
                (<span id="durationMinutes">10.0</span> min)
            </div>

            <button type="submit" class="btn">🚀 Start Inspection System</button>
            <div class="error" id="errorMsg"></div>
        </form>
    </div>

    <script>
        function getVelocityMS() {
            const v = parseFloat(document.getElementById('robotVelocity').value) || 0.6;
            const u = document.getElementById('velocityUnit').value;
            return u === 'kmh' ? v / 3.6 : v;
        }
        function updateDuration() {
            const len = parseFloat(document.getElementById('pipelineLength').value) || 100;
            const vms = getVelocityMS();
            const dur = vms > 0 ? (len / vms) : 600;
            document.getElementById('durationValue').textContent = dur.toFixed(0);
            document.getElementById('durationMinutes').textContent = (dur / 60).toFixed(1);
        }
        document.getElementById('pipelineLength').addEventListener('input', updateDuration);
        document.getElementById('robotVelocity').addEventListener('input', updateDuration);
        document.getElementById('velocityUnit').addEventListener('change', updateDuration);
        updateDuration();

        document.getElementById('configForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const len = parseFloat(document.getElementById('pipelineLength').value);
            const vms = getVelocityMS();
            const dur = vms > 0 ? (len / vms) : 600;
            const config = {
                pipeline_length: len,
                robot_velocity: vms,
                inspection_duration: dur,
                model_id: document.getElementById('modelSelect').value,
            };
            try {
                const r1 = await fetch('/api/config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                if (!r1.ok) throw new Error('Configuration failed');
                const r2 = await fetch('/api/start_project', { method: 'POST' });
                if (!r2.ok) {
                    const d = await r2.json();
                    throw new Error(d.error || 'Failed to start project');
                }
                window.location.href = '/';
            } catch (err) {
                document.getElementById('errorMsg').textContent = 'Error: ' + err.message;
                document.getElementById('errorMsg').style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Crack Detection — Pi 4</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff; padding: 20px; min-height: 100vh;
        }
        .header { text-align: center; margin-bottom: 30px; }
        h1 { color: #00ff88; font-size: 2.2em;
             text-shadow: 0 0 20px rgba(0,255,136,0.5); margin-bottom: 10px; }
        .stats-bar {
            display: flex; justify-content: center; gap: 20px;
            margin: 20px 0; flex-wrap: wrap;
        }
        .stat-item {
            background: rgba(255,255,255,0.1); padding: 10px 20px;
            border-radius: 8px; backdrop-filter: blur(10px); text-align: center;
        }
        .stat-label { font-size: 0.9em; opacity: 0.7; }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #00ff88; }

        /* Pipeline */
        .pipeline-section {
            background: rgba(255,255,255,0.05); border-radius: 15px;
            padding: 25px; margin: 20px 0; backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .pipeline-title { font-size: 1.4em; margin-bottom: 20px; color: #00ff88; }
        .pipeline-container {
            position: relative; width: 100%; height: 100px;
            background: linear-gradient(to right, #2a2a2a 0%, #3a3a3a 50%, #2a2a2a 100%);
            border-radius: 50px; overflow: visible;
            border: 3px solid #555; box-shadow: inset 0 4px 10px rgba(0,0,0,0.5);
        }
        .pipeline-markers {
            position: absolute; bottom: -28px; width: 100%;
            display: flex; justify-content: space-between;
            padding: 0 10px; font-size: 0.78em; color: #888;
        }
        .crack-marker {
            position: absolute; width: 18px; height: 18px; border-radius: 50%;
            cursor: pointer; top: 50%; transform: translate(-50%,-50%);
            transition: all 0.3s; z-index: 10; animation: pulse 2s infinite;
        }
        .crack-marker.CRITICAL { background:#ff0000; box-shadow:0 0 15px #ff0000,0 0 30px #ff000080; }
        .crack-marker.HIGH     { background:#ff6600; box-shadow:0 0 15px #ff6600,0 0 30px #ff660080; }
        .crack-marker.MEDIUM   { background:#ffff00; box-shadow:0 0 15px #ffff00,0 0 30px #ffff0080; }
        .crack-marker.LOW      { background:#00ff00; box-shadow:0 0 15px #00ff00,0 0 30px #00ff0080; }
        .crack-marker:hover { transform:translate(-50%,-50%) scale(1.5); z-index:100; }
        @keyframes pulse {
            0%,100% { transform:translate(-50%,-50%) scale(1); }
            50%      { transform:translate(-50%,-50%) scale(1.2); }
        }
        .legend {
            display:flex; justify-content:center; gap:20px;
            margin-top:20px; flex-wrap:wrap;
        }
        .legend-item { display:flex; align-items:center; gap:8px; font-size:0.9em; }
        .legend-color { width:14px; height:14px; border-radius:50%; }

        /* Modal */
        .modal {
            display:none; position:fixed; z-index:1000;
            left:0; top:0; width:100%; height:100%;
            background:rgba(0,0,0,0.8); backdrop-filter:blur(5px);
        }
        .modal-content {
            background:linear-gradient(135deg,#2a2a3e 0%,#1e2742 100%);
            margin:5% auto; border-radius:15px; width:90%; max-width:800px;
            box-shadow:0 10px 50px rgba(0,255,136,0.3);
            border:2px solid rgba(0,255,136,0.3); max-height:90vh; overflow-y:auto;
        }
        .modal-header {
            padding:20px; border-bottom:1px solid rgba(255,255,255,0.1);
            display:flex; justify-content:space-between; align-items:center;
        }
        .modal-body { padding:20px; }
        .close { color:#aaa; font-size:35px; font-weight:bold;
                 cursor:pointer; transition:color 0.3s; }
        .close:hover { color:#00ff88; }
        .crack-image { width:100%; border-radius:10px; margin-bottom:20px;
                       border:2px solid rgba(255,255,255,0.2); }
        .detail-grid {
            display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
            gap:15px; margin-top:20px;
        }
        .detail-item {
            background:rgba(255,255,255,0.05); padding:15px;
            border-radius:8px; border-left:4px solid #00ff88;
        }
        .detail-label { font-size:0.9em; opacity:0.7; margin-bottom:5px; }
        .detail-value { font-size:1.2em; font-weight:bold; }
        .severity-badge {
            display:inline-block; padding:5px 15px; border-radius:20px;
            font-weight:bold; text-transform:uppercase;
        }
        .severity-CRITICAL { background:#ff0000; color:#fff; }
        .severity-HIGH     { background:#ff6600; color:#fff; }
        .severity-MEDIUM   { background:#ffff00; color:#000; }
        .severity-LOW      { background:#00ff00; color:#000; }

        /* Camera */
        .camera-box {
            background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
            border-radius:15px; padding:20px; backdrop-filter:blur(10px);
        }
        .camera-box h2 { color:#00ff88; margin-bottom:15px; }
        .camera-box img { width:100%; border-radius:10px;
                          border:2px solid rgba(255,255,255,0.2); }
        .cam-stats {
            margin-top:15px; display:grid;
            grid-template-columns:repeat(2,1fr); gap:10px;
        }
        .cam-stat { background:rgba(0,0,0,0.3); padding:10px; border-radius:5px; }
        .cam-stat-label { font-size:0.85em; opacity:0.7; }
        .cam-stat-value { font-weight:bold; color:#00ff88; }

        /* Buttons */
        .control-btn {
            padding:12px 25px; border-radius:8px; border:none;
            background:linear-gradient(135deg,#00ff88 0%,#00cc6a 100%);
            color:#000; font-size:1.1em; font-weight:bold;
            cursor:pointer; transition:all 0.3s;
        }
        .control-btn:hover { transform:translateY(-2px);
                             box-shadow:0 5px 20px rgba(0,255,136,0.4); }
        
        /* Model Selector */
        .model-selector {
            padding:10px 20px; border-radius:8px;
            border:2px solid #00ff88; background:rgba(0,0,0,0.3);
            color:#00ff88; font-size:1em; font-weight:bold;
            cursor:pointer; transition:all 0.3s;
            margin-top:10px;
        }
        .model-selector:hover {
            background:rgba(0,255,136,0.1);
            box-shadow:0 0 15px rgba(0,255,136,0.3);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Pipeline Crack Detection System</h1>
        <p style="opacity:0.7;">Raspberry Pi 4 — Single Camera</p>
        <select class="model-selector" id="model-selector" onchange="switchModel()">
            <option value="metal">Metal Pipe Model (68% mAP)</option>
            <option value="pvc">PVC Trained Model (35% mAP)</option>
        </select>
    </div>

    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-label">Pipeline Length</div>
            <div class="stat-value" id="pipeline-length">-</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Robot Velocity</div>
            <div class="stat-value" id="robot-velocity">-</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Model</div>
            <div class="stat-value" id="model-name" style="font-size:0.8em;">-</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Current Position</div>
            <div class="stat-value" id="current-position">-</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Total Cracks</div>
            <div class="stat-value" id="total-cracks">0</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Critical</div>
            <div class="stat-value" style="color:#ff0000;" id="critical-count">0</div>
        </div>
    </div>

    <!-- Pipeline visualization -->
    <div class="pipeline-section">
        <div class="pipeline-title">Pipeline Visualization</div>
        <div class="pipeline-container" id="pipeline-container">
            <div class="pipeline-markers">
                <span>0m</span>
                <span id="pipeline-end">100m</span>
            </div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background:#ff0000;"></div><span>Critical</span></div>
            <div class="legend-item"><div class="legend-color" style="background:#ff6600;"></div><span>High</span></div>
            <div class="legend-item"><div class="legend-color" style="background:#ffff00;"></div><span>Medium</span></div>
            <div class="legend-item"><div class="legend-color" style="background:#00ff00;"></div><span>Low</span></div>
        </div>
    </div>

    <!-- Control Panel -->
    <div class="pipeline-section">
        <div class="pipeline-title">🎮 Control Panel</div>
        <div style="display:flex;gap:15px;flex-wrap:wrap;justify-content:center;margin-top:20px;">
            <button class="control-btn" id="pauseBtn" onclick="togglePause()">⏸ Pause</button>
            <button class="control-btn" onclick="stopProject()"
                    style="background:linear-gradient(135deg,#ff4444 0%,#cc0000 100%);">⏹ Stop Project</button>
            <button class="control-btn" onclick="exportPDF()"
                    style="background:linear-gradient(135deg,#4488ff 0%,#0044aa 100%);">📄 Export PDF</button>
            <button class="control-btn" onclick="window.location.reload()">🔄 Refresh</button>
        </div>
        <div id="controlStatus" style="text-align:center;margin-top:15px;font-size:1.1em;color:#00ff88;"></div>
    </div>

    <!-- Camera feed -->
    <div class="pipeline-section">
        <div class="camera-box">
            <h2>📹 Camera 0</h2>
            <img src="/video_feed" alt="Camera 0">
            <div class="cam-stats">
                <div class="cam-stat">
                    <div class="cam-stat-label">Status</div>
                    <div class="cam-stat-value" id="cam0-status">IDLE</div>
                </div>
                <div class="cam-stat">
                    <div class="cam-stat-label">Detecting</div>
                    <div class="cam-stat-value" id="cam0-detections">—</div>
                </div>
                <div class="cam-stat">
                    <div class="cam-stat-label">Total Cracks</div>
                    <div class="cam-stat-value" id="cam0-total" style="color:#ff4444;">0</div>
                </div>
                <div class="cam-stat">
                    <div class="cam-stat-label">Progress</div>
                    <div class="cam-stat-value" id="cam0-progress">0%</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Crack detail modal -->
    <div id="crackModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modal-title">Crack Details</h2>
                <span class="close">&times;</span>
            </div>
            <div class="modal-body">
                <img id="modal-image" class="crack-image" src="" alt="Crack Image">
                <div class="detail-grid">
                    <div class="detail-item"><div class="detail-label">Crack ID</div>
                        <div class="detail-value" id="modal-crack-id">-</div></div>
                    <div class="detail-item"><div class="detail-label">Position</div>
                        <div class="detail-value" id="modal-position">-</div></div>
                    <div class="detail-item"><div class="detail-label">Confidence</div>
                        <div class="detail-value" id="modal-confidence">-</div></div>
                    <div class="detail-item"><div class="detail-label">Severity</div>
                        <div class="detail-value" id="modal-severity">-</div></div>
                    <div class="detail-item"><div class="detail-label">Area (px²)</div>
                        <div class="detail-value" id="modal-area">-</div></div>
                    <div class="detail-item"><div class="detail-label">Time</div>
                        <div class="detail-value" id="modal-time">-</div></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const modal = document.getElementById('crackModal');
        document.getElementsByClassName('close')[0].onclick = () => modal.style.display = 'none';
        window.onclick = e => { if (e.target === modal) modal.style.display = 'none'; };
        const pipelineContainer = document.getElementById('pipeline-container');

        function showCrackDetail(crack) {
            document.getElementById('modal-title').textContent = `Crack #${crack.crack_id} Details`;
            document.getElementById('modal-image').src = `/crack_image/${crack.crack_id}`;
            document.getElementById('modal-crack-id').textContent = crack.crack_id;
            document.getElementById('modal-position').textContent = crack.position_m.toFixed(2) + ' m';
            document.getElementById('modal-confidence').textContent = (crack.confidence * 100).toFixed(1) + '%';
            document.getElementById('modal-area').textContent = Math.round(crack.area_px) + ' px²';
            document.getElementById('modal-time').textContent = crack.timestamp_str;
            const sev = document.getElementById('modal-severity');
            sev.innerHTML = `<span class="severity-badge severity-${crack.severity}">${crack.severity}</span>`;
            modal.style.display = 'block';
        }

        function updatePipeline() {
            fetch('/api/cracks')
                .then(r => r.json())
                .then(data => {
                    const plen = data.pipeline_length_m;
                    const cracks = data.cracks;
                    document.getElementById('pipeline-length').textContent = plen.toFixed(1) + 'm';
                    document.getElementById('pipeline-end').textContent = plen.toFixed(0) + 'm';
                    document.getElementById('current-position').textContent = data.current_position_m.toFixed(2) + 'm';
                    document.getElementById('total-cracks').textContent = cracks.length;
                    document.getElementById('cam0-total').textContent = cracks.length;

                    const prog = plen > 0 ? (data.current_position_m / plen * 100).toFixed(1) : '0.0';
                    document.getElementById('cam0-progress').textContent = prog + '%';

                    if (data.robot_velocity !== undefined) {
                        document.getElementById('robot-velocity').textContent =
                            (data.robot_velocity * 3.6).toFixed(2) + ' km/h';
                    }
                    if (data.model_id) {
                        document.getElementById('model-name').textContent =
                            data.model_id === 'offline' ? 'pipe_crack_ai' : 'Roboflow';
                    }

                    const critical = cracks.filter(c => c.severity === 'CRITICAL').length;
                    document.getElementById('critical-count').textContent = critical;

                    // Rebuild pipeline markers
                    pipelineContainer.querySelectorAll('.crack-marker').forEach(m => m.remove());
                    cracks.forEach(crack => {
                        const m = document.createElement('div');
                        m.className = `crack-marker ${crack.severity}`;
                        m.style.left = (crack.position_m / plen * 100) + '%';
                        m.title = `Crack #${crack.crack_id} @ ${crack.position_m.toFixed(2)}m — ${crack.severity}`;
                        m.onclick = () => showCrackDetail(crack);
                        pipelineContainer.appendChild(m);
                    });
                })
                .catch(e => console.error('Pipeline update error:', e));
        }

        function updateSystemStatus() {
            fetch('/api/system_status')
                .then(r => r.json())
                .then(data => {
                    if (data.cameras && data.cameras.camera0) {
                        const c = data.cameras.camera0;
                        document.getElementById('cam0-status').textContent =
                            c.status ? c.status.toUpperCase() : 'SCANNING';
                        document.getElementById('cam0-detections').textContent =
                            c.crack_detected ? '⚠ YES' : 'None';
                        document.getElementById('cam0-status').style.color =
                            c.crack_detected ? '#ff4444' : '#00ff88';
                    }
                    if (data.paused) {
                        document.getElementById('controlStatus').textContent = '⏸ PAUSED';
                        document.getElementById('controlStatus').style.color = '#ffaa00';
                        document.getElementById('pauseBtn').textContent = '▶ Resume';
                    } else if (data.started) {
                        document.getElementById('controlStatus').textContent = '▶ RUNNING';
                        document.getElementById('controlStatus').style.color = '#00ff88';
                        document.getElementById('pauseBtn').textContent = '⏸ Pause';
                    }
                })
                .catch(e => console.error('Status update error:', e));
        }

        function togglePause() {
            fetch('/api/pause_project', { method: 'POST' })
                .then(r => r.json())
                .then(d => {
                    if (d.status === 'success') updateSystemStatus();
                    else alert('Error: ' + (d.error || d.message));
                })
                .catch(e => alert('Error: ' + e));
        }

        function stopProject() {
            if (!confirm('Stop the inspection? This ends the current session.')) return;
            fetch('/api/stop_project', { method: 'POST' })
                .then(r => r.json())
                .then(d => {
                    if (d.status === 'success') {
                        alert('Project stopped. Returning to setup.');
                        window.location.reload();
                    } else {
                        alert('Error: ' + (d.error || d.message));
                    }
                })
                .catch(e => alert('Error: ' + e));
        }

        function exportPDF() {
            const s = document.getElementById('controlStatus');
            s.textContent = '📄 Generating PDF…';
            s.style.color = '#4488ff';
            window.location.href = '/api/export_pdf';
            setTimeout(() => {
                s.textContent = '✓ PDF downloaded';
                s.style.color = '#00ff88';
                setTimeout(() => { s.textContent = '▶ RUNNING'; }, 3000);
            }, 2500);
        }

        function switchModel() {
            const selector = document.getElementById('model-selector');
            const modelType = selector.value;
            
            fetch('/switch_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_type: modelType })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Switched to ' + modelType + ' model');
                } else {
                    alert('Failed to switch model: ' + data.error);
                    selector.value = data.current_model;
                }
            })
            .catch(error => {
                console.error('Error switching model:', error);
                alert('Error switching model');
            });
        }

        updatePipeline();
        updateSystemStatus();
        setInterval(updatePipeline, 2000);
        setInterval(updateSystemStatus, 1000);
    </script>
</body>
</html>
"""


def generate_frames():
    """Generate MJPEG frames for web streaming."""
    last_t = 0
    while True:
        # Throttle to ~20 FPS
        now = time.time()
        if now - last_t < 0.05:
            time.sleep(0.01)
            continue
        last_t = now

        with cam0_lock:
            frame = cam0_frame

        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            # Show a friendlier message while camera is still initializing on startup
            elapsed_since_start = time.time() - _app_start_time
            msg = "Camera Initializing..." if elapsed_since_start < 15 else "Camera Not Active"
            cv2.putText(placeholder, msg, (110, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buf = cv2.imencode('.jpg', placeholder)
        else:
            display = frame.copy()
            # Draw bounding boxes around detected cracks
            with cam0_boxes_lock:
                boxes_now = cam0_boxes if time.time() < cam0_boxes_expire else []
            for (x1, y1, x2, y2, conf) in boxes_now:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"crack {conf:.0%}"
                lx, ly = x1, max(y1 - 8, 12)
                cv2.putText(display, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            if detection_flags["cam0"]:
                cv2.putText(display, "CRACK DETECTED", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            with config_lock:
                _started = project_config.get("started", False)
                _vel = project_config.get("robot_velocity", ROBOT_VELOCITY_MPS)
                _plen = project_config.get("pipeline_length", PIPELINE_LENGTH_METERS)
            if _started:
                _elapsed = time.time() - inspection_start_time
                _pos = min(_vel * _elapsed, _plen)
                display = draw_location_indicator(display, _pos, _plen, time.strftime("%H:%M:%S"))
            ret, buf = cv2.imencode('.jpg', display)

        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


# --------------- FLASK ROUTES ----------------

@app.route('/')
def index():
    with config_lock:
        if not project_config["initialized"]:
            return render_template_string(CONFIG_TEMPLATE)
        return render_template_string(HTML_TEMPLATE)


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    global project_config, PIPELINE_LENGTH_METERS, ROBOT_VELOCITY_MPS
    global ESTIMATED_INSPECTION_DURATION_SEC, MODEL_MODE
    if request.method == 'POST':
        data = request.json
        with config_lock:
            project_config["pipeline_length"] = float(data.get('pipeline_length', 100.0))
            project_config["robot_velocity"] = float(data.get('robot_velocity', 0.167))
            project_config["inspection_duration"] = float(data.get('inspection_duration', 600.0))
            project_config["model_id"] = data.get('model_id', 'offline')
            project_config["initialized"] = True
            PIPELINE_LENGTH_METERS = project_config["pipeline_length"]
            ROBOT_VELOCITY_MPS = project_config["robot_velocity"]
            ESTIMATED_INSPECTION_DURATION_SEC = project_config["inspection_duration"]
            MODEL_MODE = project_config["model_id"]
        return jsonify({"status": "success", "message": "Configuration saved"})
    else:
        with config_lock:
            return jsonify(project_config)


@app.route('/api/start_project', methods=['POST'])
def start_project():
    global project_config, inspection_start_time, crack_log, next_id, client, local_model
    global MODEL_MODE, PIPELINE_LENGTH_METERS, ROBOT_VELOCITY_MPS
    with config_lock:
        if project_config["started"]:
            return jsonify({"error": "Project already started"}), 400

        # Reset state
        stop_event.clear()
        inspection_paused.clear()
        inspection_start_time = time.time()
        with crack_lock:
            crack_log.clear()
        next_id = 1
        detection_flags["cam0"] = False

        # Re-initialize model based on selection
        MODEL_MODE = project_config["model_id"]
        client = None
        local_model = None
        if MODEL_MODE == "online" and ROBOFLOW_AVAILABLE and RF_API_KEY:
            try:
                client = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=RF_API_KEY,
                )
                print("[WEB] Online model (Roboflow) initialized")
            except Exception as _e:
                print(f"[WEB] Failed to init online model: {_e}")
        elif MODEL_MODE == "offline" and ULTRALYTICS_AVAILABLE and Path(LOCAL_MODEL_PATH).exists():
            try:
                local_model = YOLO(LOCAL_MODEL_PATH)
                print("[WEB] Offline YOLO model initialized")
            except Exception as _e:
                print(f"[WEB] Failed to load offline model: {_e}")

        # Camera thread is started at app startup — no need to restart it here

        # Start flag updater
        def _flag_updater():
            while not stop_event.is_set():
                update_flags()
                time.sleep(0.1)
        threading.Thread(target=_flag_updater, daemon=True).start()

        project_config["started"] = True
        project_config["paused"] = False
    return jsonify({"status": "success", "message": "Project started"})


@app.route('/api/pause_project', methods=['POST'])
def pause_project():
    with config_lock:
        if not project_config["started"]:
            return jsonify({"error": "Project not started"}), 400
        project_config["paused"] = not project_config["paused"]
        if project_config["paused"]:
            inspection_paused.set()
        else:
            inspection_paused.clear()
        status = "paused" if project_config["paused"] else "resumed"
    return jsonify({"status": "success", "message": f"Project {status}"})


@app.route('/api/stop_project', methods=['POST'])
def stop_project():
    global project_config
    with config_lock:
        stop_event.set()
        inspection_paused.clear()
        project_config["started"] = False
        project_config["paused"] = False
        project_config["initialized"] = False
    return jsonify({"status": "success", "message": "Project stopped"})


@app.route('/api/system_status')
def system_status():
    with config_lock:
        cfg = dict(project_config)
    with crack_lock:
        total = len(crack_log)
    status = {
        "initialized": cfg["initialized"],
        "started": cfg["started"],
        "paused": cfg["paused"],
        "pipeline_length": cfg["pipeline_length"],
        "inspection_duration": cfg.get("inspection_duration", 0),
        "cameras": {}
    }
    if cfg["started"]:
        status["cameras"]["camera0"] = {
            "active": True,
            "status": "detecting" if detection_flags["cam0"] else "scanning",
            "crack_detected": detection_flags["cam0"],
            "confidence": 0.0,
            "count": 1 if detection_flags["cam0"] else 0,
            "total_cracks": total,
            "stats": {}
        }
    return jsonify(status)


@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch between metal pipe and PVC trained models"""
    global current_model_type
    
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'metal')
        
        # Validate model type
        if model_type not in available_models:
            return jsonify({
                'success': False,
                'error': f'Model type "{model_type}" not available',
                'current_model': current_model_type
            }), 400
        
        # Switch the model
        current_model_type = model_type
        print(f"[INFO] Switched to {model_type} model")
        
        return jsonify({
            'success': True,
            'current_model': current_model_type,
            'message': f'Successfully switched to {model_type} model'
        })
    
    except Exception as e:
        print(f"[ERROR] Model switch failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'current_model': current_model_type
        }), 500


@app.route('/api/cracks')
def api_cracks():
    with crack_lock:
        data = list(crack_log)
    cracks_out = []
    for c in data:
        cracks_out.append({
            "crack_id": c["id"],
            "camera_id": 0,
            "position_m": c.get("position_m", 0.0),
            "confidence": c["confidence"],
            "severity": c["severity"].upper(),
            "area_px": c.get("width", 0) * c.get("height", 0),
            "timestamp_str": c["timestamp"],
            "image_path": c.get("image_path", ""),
        })
    elapsed = time.time() - inspection_start_time
    current_pos = min(ROBOT_VELOCITY_MPS * elapsed, PIPELINE_LENGTH_METERS)
    with config_lock:
        vel = project_config.get("robot_velocity", ROBOT_VELOCITY_MPS)
        model_id = project_config.get("model_id", MODEL_MODE)
    return jsonify({
        "pipeline_length_m": PIPELINE_LENGTH_METERS,
        "robot_velocity": vel,
        "model_id": model_id,
        "current_position_m": current_pos,
        "total_cracks": len(cracks_out),
        "cracks": cracks_out,
    })


@app.route('/crack_image/<int:crack_id>')
def crack_image(crack_id):
    with crack_lock:
        crack = next((c for c in crack_log if c['id'] == crack_id), None)
    if crack and 'image_path' in crack and Path(crack['image_path']).exists():
        return send_file(crack['image_path'], mimetype='image/jpeg')
    return "Image not found", 404


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/export_pdf', methods=['GET'])
def export_pdf():
    try:
        with crack_lock:
            cracks_data = list(crack_log)
        report_path = REPORTS_DIR / f"inspection_report_{stamp()}.pdf"
        success = generate_pdf_report(cracks_data, report_path)
        if success and report_path.exists():
            return send_file(str(report_path), as_attachment=True,
                             download_name=report_path.name, mimetype='application/pdf')
        return "Failed to generate report", 500
    except Exception as e:
        return f"Error: {e}", 500


# ---------------- MAIN ----------------
def main():
    print(f"\n{'='*60}")
    print(f"  AUTOMATED PIPELINE INSPECTION SYSTEM — Pi 4 Optimized")
    print(f"{'='*60}")
    print(f"  Web interface : http://0.0.0.0:{FLASK_PORT}")
    print(f"  From browser  : http://<device-ip>:{FLASK_PORT}")
    print(f"\n  STEPS:")
    print(f"  1. Open the URL above in your browser")
    print(f"  2. Configure pipeline parameters on the setup page")
    print(f"  3. Click 'Start Inspection System'")
    print(f"  4. Monitor, pause, stop, or export PDF from the dashboard")
    print(f"{'='*60}\n")
    print("Press Ctrl+C to stop.\n")
    # Start camera feed immediately so the web UI shows live video from the start
    threading.Thread(target=cam_loop, args=(CAMERA_0_ID, "CAM0"), daemon=True).start()
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping system...")
        cam_stop_event.set()
        stop_event.set()
        print("[SHUTDOWN] Done.")


if __name__ == "__main__":
    main()
