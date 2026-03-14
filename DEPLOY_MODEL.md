# Deploying the YOLO Model to Raspberry Pi

## Overview
The application supports two AI models:
1. **Roboflow (Cloud)** - Online model via API, requires internet connection
2. **YOLO11n (Edge)** - Local model (68% mAP accuracy), works offline

## Deploying the YOLO Model

### Step 1: Locate Your Trained Model
On your training computer (Mac), find the model file:
```bash
~/Desktop/--/pipe_crack_ai/runs/detect/train_20260314_134701/weights/best.pt
```

### Step 2: Transfer to Raspberry Pi
Use one of these methods:

#### Option A: SCP (Secure Copy)
From your Mac terminal:
```bash
scp ~/Desktop/--/pipe_crack_ai/runs/detect/train_20260314_134701/weights/best.pt \
    shayan@192.168.8.115:/home/shayan/Desktop/Crack_App/models/best.pt
```

#### Option B: USB Drive
1. Copy `best.pt` to a USB drive on your Mac
2. Insert USB into Raspberry Pi
3. Copy from USB to Pi:
```bash
cp /media/usb/best.pt /home/shayan/Desktop/Crack_App/models/best.pt
```

#### Option C: Network Share
If you have shared folders set up between devices, copy the file through the shared location.

### Step 3: Verify Model Location
On the Raspberry Pi, confirm the model is in place:
```bash
ls -lh /home/shayan/Desktop/Crack_App/models/best.pt
```

You should see a file around 5-20 MB in size.

### Step 4: Restart the Application
The application will automatically detect the model file on startup:
```bash
cd /home/shayan/Desktop/Crack_App
python3 realtime_pi5_dual_web.py
```

Look for this message in the console:
```
✓ Offline YOLO model loaded from models/best.pt
```

## Using the Offline Model

1. Open the web interface: http://192.168.8.115:5000
2. In the configuration page, select **"Edge Model (YOLO11n - 68% mAP)"**
3. The system will run entirely offline - no internet required!

## Troubleshooting

### Model Not Found
**Error**: `ERROR: Offline model selected but model file not found`

**Solution**: Verify the model file exists:
```bash
ls -la /home/shayan/Desktop/Crack_App/models/
```

If missing, follow Steps 1-2 above to transfer the file.

### Import Error
**Error**: `WARNING: ultralytics not installed`

**Solution**: Install the package:
```bash
pip3 install ultralytics --break-system-packages
```

### Still Calling Roboflow API
**Error**: `404 Client Error: Not Found for url: https://serverless.roboflow.com/...`

**Solution**: 
1. Make sure you selected "Edge Model (YOLO11n - 68% mAP)" in the dropdown
2. Restart the application after selecting the model
3. Check that `.env` file has `LOCAL_MODEL_PATH=models/best.pt`

## Performance Notes

- **Roboflow Model**: Requires internet, ~2-3 FPS on Pi 5
- **YOLO11n Model**: Offline, ~5-8 FPS on Pi 5, lower quality (68% vs Roboflow's accuracy)

Choose based on your needs:
- Use **Roboflow** when internet is available and you need best accuracy
- Use **YOLO11n** when offline or when speed is more important than accuracy

## Model Information

### YOLO11n (pipe_crack_ai)
- **Accuracy**: 68% mAP
- **Training Date**: March 14, 2026
- **Size**: ~6 MB
- **Performance**: Faster inference on edge devices
- **Limitation**: Lower detection accuracy than Roboflow

### Roboflow Model
- **Accuracy**: Higher than YOLO11n (exact % depends on model version)
- **Connection**: Requires stable internet
- **Performance**: Slower due to network latency
- **Advantage**: Better detection quality, continuously improved
