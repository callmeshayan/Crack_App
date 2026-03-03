# Raspberry Pi 5 Dual Camera Setup

## Overview
This branch (`realtime_pi`) contains the implementation for running real-time crack detection on Raspberry Pi 5 with two cameras simultaneously.

## Hardware Requirements
- Raspberry Pi 5
- 2x USB cameras or 2x CSI cameras
- Minimum 4GB RAM recommended
- Power supply adequate for Pi 5 + cameras

## Camera Setup

### Identifying Camera Devices
On Raspberry Pi, cameras are typically available at:
- `/dev/video0`, `/dev/video1` (first camera)
- `/dev/video2`, `/dev/video3` (second camera)

To list available cameras:
```bash
v4l2-ctl --list-devices
```

### Configuration
Edit `realtime_pi5_dual.py` to set correct camera indices:
```python
CAMERA_0_INDEX = 0  # First camera
CAMERA_1_INDEX = 2  # Second camera
```

## Software Installation

### 1. System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv python3-pip v4l-utils
sudo apt-get install -y libatlas-base-dev libhdf5-dev
```

### 2. Python Environment
```bash
cd crack_app
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the example environment file and add your credentials:
```bash
cp .env.example .env
nano .env
```

Add your Roboflow credentials:
```
RF_API_KEY=your_actual_api_key
RF_WORKSPACE=your_workspace_name
RF_WORKFLOW=your_workflow_id
```

**⚠️ IMPORTANT: Never commit the `.env` file to git!**

## Running the Application

### Single Run
```bash
source .venv/bin/activate
python3 realtime_pi5_dual.py
```

### Run as Service (Background)
Create a systemd service file:
```bash
sudo nano /etc/systemd/system/crack-detection.service
```

Add:
```ini
[Unit]
Description=Crack Detection Dual Camera
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/crack_app
Environment="PATH=/home/pi/crack_app/.venv/bin"
ExecStart=/home/pi/crack_app/.venv/bin/python3 realtime_pi5_dual.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable crack-detection
sudo systemctl start crack-detection
sudo systemctl status crack-detection
```

## Performance Tuning

### Camera Resolution
Edit in `realtime_pi5_dual.py`:
```python
CAMERA_WIDTH = 640   # Lower for better performance
CAMERA_HEIGHT = 480
```

### Inference Rate
```python
INFER_FPS = 2.0  # Lower to reduce API calls and CPU usage
```

### Display Options
To run headless (no display):
```python
SHOW_PREVIEW = False
```

## Output Structure
```
data/realtime_results/
├── camera0_found/          # Camera 0 crack detections
├── camera0_realtime/       # Camera 0 live detections
├── camera1_found/          # Camera 1 crack detections
└── camera1_realtime/       # Camera 1 live detections
```

## Troubleshooting

### Camera Not Detected
```bash
# Check if cameras are recognized
ls /dev/video*

# Check camera details
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### Permission Issues
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Logout and login again
```

### Memory Issues
If running out of memory:
1. Reduce camera resolution
2. Lower INFER_FPS
3. Disable preview (SHOW_PREVIEW = False)
4. Consider using swap space

### OpenCV Issues
If cv2 import fails:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

## Key Features

- **Dual Camera Support**: Simultaneous processing of two camera feeds
- **Independent Processing**: Each camera has its own inference thread
- **Organized Outputs**: Separate folders for each camera's detections
- **Low Latency**: Optimized for Raspberry Pi 5 hardware
- **Background Operation**: Can run as a systemd service
- **Resource Efficient**: Configurable FPS and resolution

## Notes

- Uses V4L2 backend optimized for Linux/Pi
- Each camera runs independently with its own inference loop
- Frames are captured and processed asynchronously
- Detection results are saved with camera ID prefix
- ESC key stops the application (when SHOW_PREVIEW=True)
