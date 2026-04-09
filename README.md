# Automated Pipeline Inspection System

<div align="center">

**AI-Powered Real-Time Crack Detection for Industrial Pipeline Inspection**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11n-68%25%20mAP-green.svg)](https://github.com/ultralytics/ultralytics)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4%20%7C%205-red.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

The **Automated Pipeline Inspection System** is a final year engineering project developed at the University of Nicosia (2026). It provides real-time detection and classification of structural defects in industrial pipelines using a dual-camera configuration mounted on an inspection robot.

The system supports two inference modes — on-device AI (YOLOv11n) for edge computing without internet access, and a cloud-based Roboflow API for server-side processing. A live web dashboard streams annotated video feeds and generates PDF inspection reports with position-tagged defects.

Two hardware-specific implementations are provided:

- **`realtime_pi5_dual_web.py`** — Raspberry Pi 5, dual CSI cameras via `rpicam-vid`
- **`realtime_pi4_optimized_web.py`** — Raspberry Pi 4, single/dual CSI cameras via `picamera2` with `cv2.VideoCapture` fallback

---

## System Architecture

```
+-------------------------------------------------------------+
|                    OPERATOR INTERFACE                        |
|  (Pipeline Length, Robot Velocity, Model Selection)         |
+---------------------------+---------------------------------+
                            |
+---------------------------v---------------------------------+
|              AUTOMATED INSPECTION SYSTEM                    |
|                                                             |
|  +-------------------------------------------------------+  |
|  |  Raspberry Pi 4 / 5                                   |  |
|  |                                                       |  |
|  |  [ CSI Camera 0 ]         [ CSI Camera 1 ]           |  |
|  |  (1080p, 30 fps)          (1080p, 30 fps)            |  |
|  |         |                        |                   |  |
|  |         +----------+-------------+                   |  |
|  |                    |                                 |  |
|  |          [ Image Pre-processing ]                    |  |
|  |              (OpenCV)                                |  |
|  |                    |                                 |  |
|  |          [ AI Defect Detection ]                     |  |
|  |          +------------------+                        |  |
|  |          | YOLOv11n (Local) |  On-Device Mode        |  |
|  |          +------------------+                        |  |
|  |          +------------------+                        |  |
|  |          | Roboflow API     |  Cloud Mode            |  |
|  |          +------------------+                        |  |
|  |                    |                                 |  |
|  |          [ Flask Web Server - Port 5000 ]            |  |
|  +-------------------------------------------------------+  |
+---------------------------+---------------------------------+
                            |
              +-------------v--------------+
              |     Web Browser Dashboard  |
              |  - Live annotated streams  |
              |  - Detection statistics    |
              |  - Position tracking       |
              |  - PDF report export       |
              +----------------------------+
```

---

## Hardware Requirements

### Raspberry Pi 5 (Primary Target)

| Component     | Specification                               |
|---------------|---------------------------------------------|
| Model         | Raspberry Pi 5 (4 GB or 8 GB RAM)           |
| Cameras       | 2x CSI Camera Modules (15-pin ribbon cable) |
| Storage       | 32 GB+ microSD (Class 10 or faster)         |
| Power Supply  | 5 V / 5 A USB-C                             |
| Cooling       | Active cooling recommended                  |
| OS            | Raspberry Pi OS 64-bit (Bookworm)           |

### Raspberry Pi 4 (Supported)

| Component     | Specification                               |
|---------------|---------------------------------------------|
| Model         | Raspberry Pi 4 Model B (2 GB RAM minimum)  |
| Cameras       | 1x or 2x CSI Camera Modules                |
| Storage       | 32 GB+ microSD (Class 10 or faster)        |
| Power Supply  | 5 V / 3 A USB-C                            |
| OS            | Raspberry Pi OS 64-bit (Bullseye/Bookworm) |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/callmeshayan/Crack_App.git
cd Crack_App
```

### 2. Create a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
nano .env
```

### 5. Place the Trained Model

```bash
# Ensure the YOLOv11n model is present for offline mode
ls models/best.pt
```

---

## Configuration

### Interactive Startup

On launch, the system prompts the operator for the following parameters:

1. **Detection Model** — On-device (YOLOv11n) or cloud-based (Roboflow API)
2. **Pipeline Length** — Total length of the pipeline under inspection (m / km / cm)
3. **Robot Velocity** — Speed of the inspection robot (m/s, km/h, or mm/s)

### Environment Variables

All parameters can also be pre-configured via the `.env` file:

```bash
# Inference mode: offline (YOLOv11n) or online (Roboflow)
MODEL_MODE=offline

# Local model path (offline mode)
LOCAL_MODEL_PATH=models/best.pt

# Inference device: cpu, mps (Apple Silicon), or cuda
YOLO_DEVICE=cpu

# Roboflow credentials (online mode only)
RF_API_KEY=your_api_key_here
RF_WORKSPACE=your_workspace_slug
RF_WORKFLOW_ID=your_workflow_id

# Inspection parameters
PIPELINE_LENGTH=100.0
ROBOT_VELOCITY=0.5
VELOCITY_UNIT=m/s

# Camera indices
CAM0_INDEX=0
CAM1_INDEX=1
FRAME_WIDTH=1920
FRAME_HEIGHT=1080

# Flask web server
FLASK_PORT=5000
FLASK_DEBUG=False
```

---

## Running the System

### Raspberry Pi 5

```bash
python realtime_pi5_dual_web.py
```

### Raspberry Pi 4

```bash
python realtime_pi4_optimized_web.py
```

### Accessing the Web Dashboard

Once running, open a browser and navigate to:

```
http://<device-ip>:5000
```

---

## Docker Deployment

Separate Docker images are provided for each hardware target.

### Raspberry Pi 5

```bash
# Build
docker build -f Dockerfile.pi5 -t crack-app-pi5 .

# Run
docker run -it --rm \
  --privileged \
  --device /dev/video0 \
  --device /dev/video1 \
  -v /run/udev:/run/udev:ro \
  -v $(pwd)/data:/app/data \
  -p 5000:5000 \
  --env-file .env \
  crack-app-pi5
```

### Raspberry Pi 4

```bash
# Build
docker build -f Dockerfile.pi4 -t crack-app-pi4 .

# Run
docker run -it --rm \
  --privileged \
  --device /dev/video0 \
  -v /run/udev:/run/udev:ro \
  -v $(pwd)/data:/app/data \
  -p 5000:5000 \
  --env-file .env \
  crack-app-pi4
```

---

## Web Dashboard and API

| Endpoint              | Method | Description                         |
|-----------------------|--------|-------------------------------------|
| `/`                   | GET    | Main monitoring dashboard           |
| `/video_feed/0`       | GET    | Camera 0 live MJPEG stream          |
| `/video_feed/1`       | GET    | Camera 1 live MJPEG stream          |
| `/api/cracks`         | GET    | JSON list of all detected defects   |
| `/generate_report`    | GET    | Generate and download PDF report    |
| `/crack_image/<id>`   | GET    | Retrieve a saved crack image by ID  |

---

## Detection Modes

### On-Device AI (YOLOv11n)

- Operates entirely offline — no internet connection required
- Model: YOLOv11n, 68% mAP, approximately 5.5 MB
- Real-time inference on Raspberry Pi 5 at 15-20 FPS
- Supports CPU, Apple Silicon (MPS), and CUDA targets

### Cloud-Based AI (Roboflow API)

- Requires internet access and a valid Roboflow API key
- Server-side inference with automatic model versioning
- Typical latency: 150-300 ms per frame

---

## Performance Summary

| Metric              | On-Device (Pi 5) | Cloud (Roboflow) |
|---------------------|------------------|------------------|
| Inference Speed     | 15-20 FPS        | 10-15 FPS        |
| Latency             | 50-70 ms         | 150-300 ms       |
| Accuracy (mAP)      | 68%              | Varies by model  |
| Internet Required   | No               | Yes              |
| Estimated Power Draw| 5-8 W            | 3-5 W            |

---

## Project Structure

```
Crack_App/
├── realtime_pi5_dual_web.py       # Main application — Raspberry Pi 5
├── realtime_pi4_optimized_web.py  # Main application — Raspberry Pi 4
├── requirements.txt               # Python package dependencies
├── .env.example                   # Environment configuration template
├── .python-version                # Python version pin (3.11)
├── Dockerfile.pi5                 # Docker image for Raspberry Pi 5
├── Dockerfile.pi4                 # Docker image for Raspberry Pi 4
├── models/
│   └── best.pt                    # Trained YOLOv11n model (offline mode)
└── data/                          # Runtime output (generated at startup)
    └── realtime_results/
        ├── reports/               # PDF and CSV inspection reports
        ├── cam0_found/            # Crack images from Camera 0
        └── cam1_found/            # Crack images from Camera 1
```

---

## Troubleshooting

### Camera Not Detected

```bash
# List available cameras
libcamera-hello --list-cameras

# Test a single capture
libcamera-jpeg -o test.jpg --camera 0
```

### Model File Missing

```bash
# Verify the model file is present
ls -lh models/best.pt
```

### Port Already in Use

```bash
# Identify the process occupying port 5000
lsof -i :5000

# Terminate it
kill -9 <PID>
```

### Insufficient Memory on Raspberry Pi 4

```bash
# Increase swap space to 2 GB
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors

**Final Year Engineering Project — University of Nicosia, 2026**

Shayan Naghashpour Shoushtari
Contact: nshayan81@hotmail.com

---

## Acknowledgments

- Ultralytics — YOLOv11 object detection framework
- Roboflow — Cloud computer vision platform
- Raspberry Pi Foundation — Hardware platform
- Flask — Lightweight Python web framework
- OpenCV — Open-source computer vision library
