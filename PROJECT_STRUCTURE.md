# Project Structure - Automated Pipeline Inspection System

## 📁 Repository Organization

```
crack_app/
│
├── 🎯 Core Application Files
│   ├── realtime_pi5_dual_web.py      # Main application: Real-time dual camera inspection with web UI (Pi 5)
│   ├── realtime_pi4_optimized_web.py # Optimized real-time inspection with web UI (Pi 4)
│   ├── pipeline_inspection.py         # Core inspection logic and pipeline position tracking
│   ├── gui_app.py                     # Alternative GUI interface for desktop use
│   └── webapp.py                      # Standalone web server for remote monitoring
│
├── 🔍 Inference & Processing Scripts
│   ├── infer_image.py                 # Single image inference
│   ├── infer_image_workflow.py        # Image inference with full workflow
│   ├── batch_infer_workflow.py        # Batch processing workflow
│   └── batch_process_images.py        # Batch image processing utility
│
├── 🤖 Model & Detection
│   ├── models/                        # YOLOv11n model files (.pt)
│   └── roboflow_client.py            # Roboflow API integration for cloud inference
│
├── 🌐 Web Interface
│   └── templates/                     # HTML templates for web dashboard
│       └── index.html                # Main monitoring interface
│
├── 📊 Data & Results
│   ├── data/                         # Input images and processing data
│   │   ├── batch_images/            # Batch processing input
│   │   └── results/                 # Generated inspection reports (PDFs)
│   ├── images/                       # Sample/test images
│   ├── images_batch/                 # Batch processing samples
│   ├── outputs_batch/                # Batch processing outputs
│   └── outputs_model/                # Model inference outputs
│
├── 📝 Documentation
│   ├── README.md                     # Main project documentation (537 lines)
│   ├── OPERATOR_GUIDE.md             # Operator instructions
│   ├── ARCHITECTURE_DIAGRAM.txt      # System architecture overview
│   ├── MODEL_MODE_REFERENCE.md       # AI model configuration guide
│   ├── INTEGRATION_SUMMARY.md        # System integration details
│   ├── OFFLINE_MODEL_SETUP.md        # Offline/edge deployment guide
│   ├── DEPLOY_MODEL.md               # Model deployment instructions
│   ├── README_RASPBERRY_PI.md        # Raspberry Pi specific setup
│   ├── QUICK_REFERENCE.txt           # Quick command reference
│   └── STARTUP_EXAMPLE.txt           # Startup configuration examples
│
├── 🚀 Deployment & Testing
│   ├── Dockerfile                     # Container configuration
│   ├── deploy_model_to_pi.sh         # Model deployment script
│   ├── run_docker.sh                 # Docker execution script
│   ├── run_image_inference.sh        # Image inference runner
│   ├── run_realtime.sh               # Real-time system launcher
│   ├── smoke_test.py                 # System smoke tests
│   └── test_offline_mode.py          # Offline mode testing
│
├── 🔧 Configuration
│   ├── requirements.txt               # Python dependencies
│   ├── .env                          # Environment variables (Roboflow API key)
│   ├── .env.example                  # Environment template
│   ├── .gitignore                    # Git ignore rules
│   └── .python-version               # Python version specification
│
├── 📦 Archive
│   └── archive/                       # Older script versions
│       ├── realtime.py               # Legacy realtime version
│       └── realtime_pi5_dual.py      # Early Pi 5 dual camera version (non-web)
│
└── 🔨 Development
    └── marker.py                      # Development utility script
```

## 🎓 Key Components for Presentation

### 1. Main Application
**File**: [realtime_pi5_dual_web.py](realtime_pi5_dual_web.py)
- Real-time crack detection using dual CSI cameras on Raspberry Pi 5
- Web-based monitoring dashboard (port 5000)
- Velocity-based position tracking for accurate crack location
- Dual AI modes: On-device YOLOv11n + Cloud Roboflow API
- Professional PDF report generation with severity analysis

### 2. Core Features
- **Dual Camera Support**: Simultaneous monitoring from two cameras
- **Real-Time Processing**: Live video with detection overlays
- **Position Tracking**: Physics-based crack location (m/s or km/h)
- **Edge Computing**: Runs offline on Raspberry Pi without internet
- **Web Interface**: Browser-based monitoring and control

### 3. AI Models
- **On-Device**: YOLOv11n (68% mAP) - Fast, edge-optimized
- **Cloud**: Roboflow API - High accuracy when internet available
- **Dual Mode**: Automatic fallback for reliability

### 4. Documentation Highlights
- **README.md**: Comprehensive 537-line documentation
- **OPERATOR_GUIDE.md**: User-friendly operation instructions
- **ARCHITECTURE_DIAGRAM.txt**: System design overview

## 📊 Technical Specifications

### Hardware Requirements
- **Platform**: Raspberry Pi 5 (8GB RAM recommended)
- **Cameras**: 2x CSI cameras (1080p @ 30fps)
- **Storage**: microSD card (32GB+)
- **Optional**: Cooling fan for sustained operation

### Software Stack
- **Language**: Python 3.9+
- **Framework**: YOLOv11 (Ultralytics)
- **Web**: Flask
- **Vision**: OpenCV, Picamera2
- **ML**: PyTorch

### Performance Metrics
- **Inference Speed**: ~15-30 FPS per camera (on-device)
- **Detection Accuracy**: 68% mAP (YOLOv11n)
- **Latency**: <100ms per frame
- **Network**: Runs fully offline (optional cloud mode)

## 🎯 For Presentation Demo

### Quick Start Command
```bash
python realtime_pi5_dual_web.py
```

### Demo Flow
1. **Setup**: Enter pipeline parameters (length, velocity)
2. **Model Selection**: Choose on-device or cloud mode
3. **Camera Check**: Verify dual camera feeds
4. **Live Monitoring**: View real-time detection on web dashboard
5. **Report**: Generate PDF inspection report with findings

### Key Selling Points
✅ **Production Ready**: Fully functional edge computing system
✅ **Dual Redundancy**: Two cameras + two AI models
✅ **Professional Output**: Automated PDF reports
✅ **User Friendly**: Web interface + interactive setup
✅ **Scalable**: Docker support, modular architecture

## 📈 Repository Statistics

- **Total Lines of Code**: ~2,500+ lines
- **Documentation Pages**: 8 comprehensive guides
- **Python Modules**: 15+ modular scripts
- **Disk Usage**: ~5.5MB (without data/venv)
- **Saved Space**: 2.2GB cleaned (removed redundant files)

---

**Last Updated**: March 23, 2026
**Version**: Production Release
**License**: MIT
