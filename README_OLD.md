# Crack Detection Application

Realtime crack detection system using Roboflow API and computer vision.

## Features
- Real-time camera-based crack detection
- GUI application with PySide6
- Batch image processing
- Configurable confidence thresholds
- Automatic image saving and classification

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install opencv-python python-dotenv inference-sdk PySide6 numpy requests
```

3. Create a `.env` file with your Roboflow credentials:
```
RF_API_KEY=your_api_key
RF_WORKSPACE=your_workspace
RF_WORKFLOW_ID=your_workflow_id
RF_MODEL_ID=1
RF_CONF=0.5
```

## Usage

### GUI Application
```bash
python gui_app.py
```

### Realtime Detection (Terminal)
```bash
python realtime.py
```

### Single Image Inference
```bash
python infer_image.py path/to/image.jpg
```

### Batch Processing
```bash
python batch_infer_workflow.py
```

## Files
- `gui_app.py` - PySide6 GUI application
- `realtime.py` - Terminal-based realtime detection
- `batch_infer_workflow.py` - Batch image processing
- `infer_image.py` - Single image inference
- `roboflow_client.py` - Roboflow API client
- `marker.py` - Marker utility class

## Requirements
- Python 3.8+
- OpenCV
- PySide6
- Roboflow Inference SDK
