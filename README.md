# Automated Pipeline Inspection System

<div align="center">

**AI-Powered Real-Time Crack Detection for Industrial Pipeline Inspection**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11n-68%25%20mAP-green.svg)](https://github.com/ultralytics/ultralytics)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-red.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🎯 Overview

The **Automated Pipeline Inspection System** is an advanced industrial solution for real-time detection and monitoring of pipeline defects using dual-camera configuration and AI-powered computer vision. The system provides both edge computing (on-device) and cloud-based detection modes for maximum flexibility.

### Key Capabilities
- ✅ **Dual AI Models**: On-Device YOLOv11n (68% mAP) + Cloud-Based Roboflow API
- ✅ **Real-Time Processing**: Live video streaming with annotated detection overlay
- ✅ **Dual Camera Support**: Simultaneous monitoring from two CSI cameras (Raspberry Pi 5)
- ✅ **Velocity-Based Positioning**: Physics-based crack location tracking (m/s or km/h)
- ✅ **Web Interface**: Browser-based monitoring dashboard at port 5000
- ✅ **PDF Report Generation**: Professional inspection reports with severity analysis
- ✅ **Edge Computing**: Runs offline on Raspberry Pi 5 without internet
- ✅ **Interactive Configuration**: Operator-friendly startup prompts
- ✅ **Production Ready**: Docker support, comprehensive logging, error handling

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OPERATOR INTERFACE                        │
│  (Pipeline Length, Velocity, Model Selection)                │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│              AUTOMATED INSPECTION SYSTEM                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Raspberry Pi 5 (8GB RAM)                            │    │
│  │  ┌─────────────┐         ┌─────────────┐            │    │
│  │  │  CSI CAM0   │         │  CSI CAM1   │            │    │
│  │  │ (1080p 30fps)         │ (1080p 30fps)            │    │
│  │  └──────┬──────┘         └──────┬──────┘            │    │
│  │         │                       │                    │    │
│  │         └───────┬───────────────┘                    │    │
│  │                 │                                    │    │
│  │         ┌───────▼─────────┐                          │    │
│  │         │ Image Processing │                          │    │
│  │         │   (OpenCV)       │                          │    │
│  │         └───────┬─────────┘                          │    │
│  │                 │                                    │    │
│  │         ┌───────▼─────────┐                          │    │
│  │         │  AI Detection    │                          │    │
│  │         │  ┌────────────┐  │                          │    │
│  │         │  │ YOLOv11n   │  │ (On-Device)             │    │
│  │         │  │ 68% mAP    │  │                          │    │
│  │         │  └────────────┘  │                          │    │
│  │         │  ┌────────────┐  │                          │    │
│  │         │  │ Roboflow   │  │ (Cloud API)             │    │
│  │         │  │    API     │  │                          │    │
│  │         │  └────────────┘  │                          │    │
│  │         └───────┬─────────┘                          │    │
│  │                 │                                    │    │
│  │         ┌───────▼─────────┐                          │    │
│  │         │ Flask Web Server │                          │    │
│  │         │   (Port 5000)    │                          │    │
│  │         └───────┬─────────┘                          │    │
│  └─────────────────┼─────────────────────────────────────┘    │
└────────────────────┼─────────────────────────────────────────┘
                     │
         ┌───────────▼────────────┐
         │  Web Browser Dashboard  │
         │  • Live Video Streams   │
         │  • Detection Statistics │
         │  • Position Tracking    │
         │  • PDF Report Export    │
         └────────────────────────┘
```

---

## 🚀 Quick Start

### For Raspberry Pi 5 (Production)

1. **Clone the repository**
```bash
cd ~/Desktop
git clone <repository-url> Crack_App
cd Crack_App
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
nano .env  # Edit with your settings
```

5. **Deploy AI model** (for offline mode)
```bash
# Copy your trained model to models/ directory
cp /path/to/your/best.pt models/best.pt
```

6. **Run the system**
```bash
python realtime_pi5_dual_web.py
```

7. **Access web interface**
   - Open browser: `http://raspberrypi-ip:5000`
   - Or locally: `http://localhost:5000`

### For Development (Mac/Linux)

```bash
# Clone and setup
git clone <repository-url> crack_app
cd crack_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env

# Run with test images (no camera required)
python realtime_pi5_dual_web.py
```

### Using Docker

```bash
# Build image
docker build -t pipeline-inspection .

# Run container
./run_docker.sh
```

---

## 📋 Configuration

### Interactive Startup
When you run the system, you'll be prompted to configure:

1. **Detection Model Selection**
   - Option 1: On-Device AI Model (YOLOv11n - 68% mAP, Edge Computing)
   - Option 2: Cloud-Based AI Model (Roboflow API - Real-time Processing)

2. **Pipeline Length** (meters)
   - Total length of the pipeline to inspect

3. **Robot Velocity** (m/s or km/h)
   - Speed of the inspection robot/camera

### Environment Variables (.env)

```bash
# Model Configuration
MODEL_MODE=offline              # offline (YOLOv11n) or online (Roboflow)
LOCAL_MODEL_PATH=models/best.pt # Path to YOLO model
YOLO_DEVICE=cpu                 # cpu, mps, or cuda

# Roboflow API (for online mode)
RF_API_KEY=your_api_key
RF_WORKSPACE=your_workspace
RF_WORKFLOW_ID=your_workflow_id

# Inspection Parameters
ROBOT_VELOCITY=0.5              # Robot speed
VELOCITY_UNIT=m/s               # m/s or km/h
PIPELINE_LENGTH=100.0           # Pipeline length in meters

# Camera Settings
CAM0_INDEX=0                    # Primary camera
CAM1_INDEX=1                    # Secondary camera
FRAME_WIDTH=1920
FRAME_HEIGHT=1080

# Web Server
FLASK_PORT=5000
FLASK_DEBUG=False
```

---

## 🎮 Features in Detail

### 1. Dual AI Detection Modes

#### On-Device AI (YOLOv11n)
- **Offline Operation**: No internet required
- **Performance**: 68% mAP (mean Average Precision)
- **Speed**: Real-time inference on Raspberry Pi 5
- **Hardware Support**: CPU, Apple Silicon (MPS), CUDA GPU
- **Model Size**: ~5.5 MB

#### Cloud-Based AI (Roboflow API)
- **Online Service**: Requires internet connection
- **Scalability**: Server-side processing
- **Flexibility**: Easy model updates without device access
- **API Rate Limits**: Check Roboflow pricing

### 2. Real-Time Web Dashboard

Access at `http://raspberrypi-ip:5000`

**Features:**
- Live video streams from both cameras
- Real-time detection annotations
- Current position tracking along pipeline
- Crack detection statistics
- Severity distribution charts
- Inspection timeline
- PDF report generation button

### 3. Position Tracking System

**Physics-Based Calculation:**
```
Position (m) = Velocity (m/s) × Time Elapsed (s)
Progress (%) = (Current Position / Total Length) × 100
```

**Visual Indicators:**
- Position overlay on saved crack images
- Progress bar showing pipeline coverage
- Distance remaining calculation
- Estimated completion time

### 4. PDF Report Generation

**Automated Reports Include:**
- Inspection summary and parameters
- Total cracks detected by severity
- Detailed crack list with timestamps
- Position information for each defect
- System configuration details
- Professional formatting with charts

**Generate Reports:**
- Via web dashboard: Click "Generate PDF Report"
- REST API: `GET /generate_report`

---

## 📁 Project Structure

```
crack_app/
├── realtime_pi5_dual_web.py   # Main application (Raspberry Pi 5)
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── Dockerfile                 # Container configuration
├── run_docker.sh             # Docker run script
│
├── models/                    # AI model files
│   └── best.pt               # YOLOv11n trained model
│
├── templates/                 # Web interface templates
│   └── (Flask HTML templates)
│
├── data/                      # Runtime data
│   ├── inspection_log.csv    # Detection logs
│   └── crack_images/         # Saved crack images
│
├── outputs_batch/            # Batch processing results
├── images_batch/             # Batch input images
│
├── Documentation/
│   ├── DEPLOY_MODEL.md       # Model deployment guide
│   ├── OPERATOR_GUIDE.md     # User manual
│   ├── ARCHITECTURE_DIAGRAM.txt
│   ├── INTEGRATION_SUMMARY.md
│   ├── MODEL_MODE_REFERENCE.md
│   └── OFFLINE_MODEL_SETUP.md
│
└── Utilities/
    ├── gui_app.py            # Desktop GUI application
    ├── batch_infer_workflow.py
    ├── infer_image.py        # Single image testing
    ├── test_offline_mode.py  # Model testing
    └── smoke_test.py         # System validation
```

---

## 🔧 Hardware Requirements

### Raspberry Pi 5 (Recommended)

| Component | Specification |
|-----------|--------------|
| **Model** | Raspberry Pi 5 (4GB or 8GB RAM) |
| **Cameras** | 2× CSI Camera Modules (15-pin ribbon) |
| **Storage** | 32GB+ microSD (Class 10 or better) |
| **Power** | 5V 5A USB-C Power Supply |
| **Cooling** | Active cooling fan (recommended) |
| **OS** | Raspberry Pi OS (64-bit) Bookworm |

### Development Machine

| Component | Requirement |
|-----------|------------|
| **OS** | macOS, Linux, or Windows |
| **Python** | 3.9 - 3.13 |
| **RAM** | 8GB+ |
| **Storage** | 10GB+ free space |

---

## 🎓 Usage Examples

### Example 1: Basic Inspection

```bash
# Start system
python realtime_pi5_dual_web.py

# Follow prompts:
# Select Model: 1 (On-Device AI)
# Pipeline Length: 50 meters
# Velocity: 0.5 m/s

# Access dashboard
# Open browser: http://192.168.1.100:5000
```

### Example 2: High-Speed Cloud Processing

```bash
# Configure for cloud mode
export MODEL_MODE=online

# Start system
python realtime_pi5_dual_web.py

# Select Model: 2 (Cloud-Based AI)
# Pipeline Length: 200 meters
# Velocity: 2.5 km/h
```

### Example 3: Batch Image Processing

```bash
# Process existing images
python batch_infer_workflow.py

# Results saved in outputs_batch/
```

### Example 4: Single Image Testing

```bash
# Test model on single image
python infer_image.py test_cam0.jpg

# View annotated result
```

---

## 🌐 API Endpoints

### Web Dashboard
- `GET /` - Main dashboard page

### Detection Data
- `GET /api/cracks` - JSON list of all detected cracks

### Video Streams
- `GET /video_feed/0` - CAM0 live stream (MJPEG)
- `GET /video_feed/1` - CAM1 live stream (MJPEG)

### Reports
- `GET /generate_report` - Generate and download PDF report

### Images
- `GET /crack_image/<id>` - Retrieve saved crack image by ID

---

## 🐛 Troubleshooting

### Camera Issues
```bash
# Check camera detection
libcamera-hello --list-cameras

# Test camera capture
libcamera-jpeg -o test.jpg --camera 0
```

### Model Loading Errors
```bash
# Verify model exists
ls -lh models/best.pt

# Test model inference
python test_offline_mode.py
```

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>
```

### Python Version Issues
```bash
# Check Python version (requires 3.9-3.13)
python --version

# Use specific version
python3.11 -m venv venv
```

### Memory Errors on Raspberry Pi
```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## 📊 Performance Metrics

| Metric | On-Device (Pi 5) | Cloud-Based |
|--------|------------------|-------------|
| **Inference Speed** | ~15-20 FPS | ~10-15 FPS |
| **Latency** | 50-70ms | 150-300ms |
| **Accuracy (mAP)** | 68% | Varies |
| **Internet Required** | ❌ No | ✅ Yes |
| **Power Consumption** | 5-8W | 3-5W |
| **Model Updates** | Manual | Automatic |

---

## 🛠️ Development

### Running Tests

```bash
# Smoke test (system validation)
python smoke_test.py

# Offline mode test
python test_offline_mode.py

# Camera test
python realtime_pi5_dual.py
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

---

## 📚 Documentation

- [Model Deployment Guide](DEPLOY_MODEL.md)
- [Operator Manual](OPERATOR_GUIDE.md)
- [Architecture Overview](ARCHITECTURE_DIAGRAM.txt)
- [Integration Summary](INTEGRATION_SUMMARY.md)
- [Offline Mode Setup](OFFLINE_MODEL_SETUP.md)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Final Year Project 2026**
- Industrial Pipeline Inspection System
- AI-Powered Defect Detection

---

## 🙏 Acknowledgments

- **Ultralytics YOLOv11** - Object detection framework
- **Roboflow** - Cloud-based computer vision platform
- **Raspberry Pi Foundation** - Hardware platform
- **Flask** - Web framework
- **OpenCV** - Computer vision library

---

## 📞 Support

For issues, questions, or contributions:
- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Docs: [Project Wiki](https://github.com/your-repo/wiki)

---

<div align="center">

**Built with ❤️ for industrial automation and AI-powered inspection**

[⬆ Back to top](#automated-pipeline-inspection-system)

</div>
