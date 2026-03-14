# Offline Model Setup Guide

This guide explains how to integrate and use your locally trained YOLO model on the Raspberry Pi for crack detection.

## Overview

The crack detection app now supports two modes:
- **Online Mode**: Uses Roboflow Rapid API (requires internet connection)
- **Offline Mode**: Uses your locally trained YOLO model (works without internet)

## Setup Instructions

### 1. Copy Your Model to Raspberry Pi

Transfer your trained model weights to the Raspberry Pi:

```bash
# From your computer, copy the model to the Pi
scp /Users/shayannaghashpour/Desktop/--/pipe_crack_ai/runs/detect/train_20260314_134701/weights/best.pt \
    pi@raspberrypi.local:~/crack_app/models/
```

### 2. Configure the .env File

On the Raspberry Pi, edit `.env`:

```bash
nano ~/crack_app/.env
```

Update the following settings:

```env
# Choose your mode
MODEL_MODE=offline

# Path to your model on Raspberry Pi
LOCAL_MODEL_PATH=/home/pi/crack_app/models/best.pt
```

### 3. Install Dependencies

On the Raspberry Pi:

```bash
cd ~/crack_app

# Install ultralytics and PyTorch for CPU
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or if you have all requirements
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python realtime_pi5_dual_web.py
```

The app will automatically use your offline model!

## Switching Between Modes

### To Use Offline Mode (Local YOLO Model)

Edit `.env`:
```env
MODEL_MODE=offline
LOCAL_MODEL_PATH=/home/pi/crack_app/models/best.pt
```

Benefits:
- ✓ No internet required
- ✓ Faster inference (no API calls)
- ✓ No API costs
- ✓ Full control over model

### To Use Online Mode (Roboflow API)

Edit `.env`:
```env
MODEL_MODE=online
RF_API_KEY=your_api_key
RF_WORKSPACE=your_workspace
RF_WORKFLOW_ID=your_workflow
```

Benefits:
- ✓ No local model storage needed
- ✓ Easy model updates via Roboflow
- ✓ Access to Roboflow's preprocessing pipeline

## Performance Optimization for Raspberry Pi

### Model Size
Your current model is YOLO11n (nano), which is already optimized for edge devices like Raspberry Pi.

### Inference Settings

In `.env`, adjust these for better performance:

```env
# Lower confidence threshold for more detections
RF_CONF=0.3

# Reduce FPS if Pi is struggling
RF_INFER_FPS=5.0

# Camera resolution (in code, default 640x480 is good for Pi)
```

### CPU vs GPU
The code defaults to `device='cpu'` which is correct for Raspberry Pi 5. If you ever run on a system with CUDA GPU, you can modify the code to use `device='0'`.

## Troubleshooting

### "Model file not found"
- Check the path in `.env` matches where you copied the model
- Use absolute paths, not relative

### "ultralytics not installed"
```bash
pip install ultralytics torch torchvision
```

### Slow inference on Raspberry Pi
- Reduce camera resolution in code (CAMERA_WIDTH, CAMERA_HEIGHT)
- Lower RF_INFER_FPS in .env
- Consider using YOLO11n (nano) instead of larger models

### Out of memory errors
- Close other applications on Pi
- Reduce batch size (already set to 1 for single image inference)
- Consider adding swap space

## Model Information

**Current Model**: `/pipe_crack_ai/runs/detect/train_20260314_134701/weights/best.pt`

This model was trained on your custom crack detection dataset and optimized for pipeline inspection.

### Model Performance
Check your training results in:
```
/pipe_crack_ai/runs/detect/train_20260314_134701/
```

Look at:
- `results.png` - Training metrics
- `confusion_matrix.png` - Model accuracy
- `val_batch*.jpg` - Validation predictions

## Deployment Checklist

- [ ] Copy model weights to Raspberry Pi
- [ ] Update LOCAL_MODEL_PATH in .env
- [ ] Set MODEL_MODE=offline
- [ ] Install ultralytics and torch
- [ ] Test inference with sample images
- [ ] Verify camera connections
- [ ] Test end-to-end pipeline

## Next Steps

1. **Test the model** on real pipe images from your cameras
2. **Monitor performance** - FPS, detection accuracy
3. **Adjust confidence threshold** based on false positive/negative rate
4. **Retrain if needed** with more diverse data

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all paths in .env are correct
3. Ensure model file is not corrupted (check file size)
4. Test with both online and offline modes to isolate issues
