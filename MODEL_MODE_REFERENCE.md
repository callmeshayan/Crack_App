# Quick Reference: Model Mode Selection

## Current Configuration

**Default Mode**: Offline (Local YOLO Model)

## Fast Mode Switching

Edit `.env` file and change `MODEL_MODE`:

### Option 1: Offline Mode (Recommended for Pi)
```env
MODEL_MODE=offline
LOCAL_MODEL_PATH=/home/pi/crack_app/models/best.pt
```

### Option 2: Online Mode (Roboflow API)
```env
MODEL_MODE=online
RF_API_KEY=yJllSfiKmS9kxebNGvUP
RF_WORKSPACE=shayanfyp
RF_WORKFLOW_ID=find-cracks-2
```

## Model Comparison

| Feature | Offline Mode | Online Mode |
|---------|--------------|-------------|
| Internet Required | ❌ No | ✅ Yes |
| Inference Speed | ⚡ Fast (local) | 🐌 Slower (API) |
| Setup Complexity | Medium | Easy |
| Cost | Free | API credits |
| Customization | Full control | Limited |
| Model Updates | Manual | Automatic |
| Best For | Production Pi | Development/Testing |

## Commands

### Deploy to Raspberry Pi
```bash
./deploy_model_to_pi.sh raspberrypi.local pi
```

### Test Model Locally
```bash
cd ~/crack_app
python realtime_pi5_dual_web.py
```

### Check Model Info
```bash
# On Mac
ls -lh /Users/shayannaghashpour/Desktop/--/pipe_crack_ai/runs/detect/*/weights/best.pt

# On Pi
ls -lh ~/crack_app/models/best.pt
```

## Troubleshooting Quick Fixes

**Problem**: Model not found
```bash
# Check path in .env matches actual file location
ls -l $(grep LOCAL_MODEL_PATH .env | cut -d= -f2)
```

**Problem**: Slow inference
```env
# In .env, reduce FPS
RF_INFER_FPS=3.0
```

**Problem**: Import errors
```bash
pip install ultralytics torch torchvision --upgrade
```

## Performance Tips

1. **Offline mode** is 2-3x faster than online mode
2. Start with **RF_CONF=0.3** and adjust based on false positives
3. **YOLO11n** (nano) model is optimal for Raspberry Pi 5
4. Monitor Pi temperature during long runs: `vcgencmd measure_temp`

## Files Modified

- ✅ `realtime_pi5_dual_web.py` - Added dual-mode inference
- ✅ `requirements.txt` - Added ultralytics dependency
- ✅ `.env` - Added MODEL_MODE configuration
- ✅ `OFFLINE_MODEL_SETUP.md` - Full setup guide
- ✅ `deploy_model_to_pi.sh` - Automated deployment script
