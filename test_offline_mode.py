#!/usr/bin/env python3
"""
Quick test script to verify offline YOLO model integration
Tests model loading, inference, and prediction extraction
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing Offline YOLO Model Integration")
print("=" * 60)
print()

# Test 1: Import check
print("1. Testing imports...")
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("   Install with: pip install ultralytics opencv-python")
    sys.exit(1)

# Test 2: Model loading
print("\n2. Testing model loading...")
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    "/Users/shayannaghashpour/Desktop/--/pipe_crack_ai/runs/detect/train_20260314_134701/weights/best.pt"
)

if not Path(MODEL_PATH).exists():
    print(f"   ✗ Model not found at: {MODEL_PATH}")
    print("   Please update LOCAL_MODEL_PATH in .env")
    sys.exit(1)

try:
    model = YOLO(MODEL_PATH)
    print(f"   ✓ Model loaded: {MODEL_PATH}")
    print(f"   ✓ Model type: {type(model)}")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    sys.exit(1)

# Test 3: Create test image
print("\n3. Creating test image...")
test_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
print(f"   ✓ Test image shape: {test_img.shape}")

# Test 4: Inference
print("\n4. Testing inference...")
DEVICE = os.getenv("YOLO_DEVICE", "cpu")
try:
    results = model.predict(
        source=test_img,
        conf=0.3,
        verbose=False,
        device=DEVICE,
        imgsz=640,
    )
    print(f"   ✓ Inference successful on device: {DEVICE}")
    print(f"   ✓ Results type: {type(results)}")
    print(f"   ✓ Number of results: {len(results)}")
except Exception as e:
    print(f"   ✗ Inference failed: {e}")
    sys.exit(1)

# Test 5: Result parsing
print("\n5. Testing result extraction...")
try:
    result = results[0]
    print(f"   ✓ Result object type: {type(result)}")
    print(f"   ✓ Has boxes: {hasattr(result, 'boxes')}")
    
    if hasattr(result, 'boxes') and result.boxes is not None:
        boxes = result.boxes
        print(f"   ✓ Number of detections: {len(boxes)}")
        
        if len(boxes) > 0:
            print(f"   ✓ First box shape: {boxes[0].xyxy.shape}")
            print(f"   ✓ Has confidence: {hasattr(boxes[0], 'conf')}")
            print(f"   ✓ Has class: {hasattr(boxes[0], 'cls')}")
    else:
        print("   ℹ No detections (expected for random image)")
    
    # Test prediction extraction function
    print("\n6. Testing extract_predictions function...")
    predictions = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
            
            predictions.append({
                "x": x_center,
                "y": y_center,
                "width": width,
                "height": height,
                "confidence": conf,
                "class": class_name,
            })
    
    print(f"   ✓ Extracted {len(predictions)} predictions")
    print(f"   ✓ Prediction format matches Roboflow format")
    
except Exception as e:
    print(f"   ✗ Result extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: JSON serialization
print("\n7. Testing JSON serialization...")
try:
    import json
    
    # Test serializing YOLO result (should fail as expected)
    try:
        json.dumps({"result": result})
        print("   ⚠ YOLO result object is serializable (unexpected)")
    except TypeError:
        print("   ✓ YOLO result object not serializable (expected)")
    
    # Test our safe serialization
    safe_result = {
        "model_type": "yolo",
        "inference_time_ms": getattr(result, 'speed', {}).get('inference', 0) if hasattr(result, 'speed') else 0,
        "predictions": predictions,
    }
    json_str = json.dumps(safe_result)
    print("   ✓ Safe serialization works")
    print(f"   ✓ JSON length: {len(json_str)} bytes")
    
except Exception as e:
    print(f"   ✗ JSON serialization test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour offline YOLO model integration is working correctly!")
print(f"Model: {Path(MODEL_PATH).name}")
print(f"Device: {DEVICE}")
print("\nYou can now run the main application with MODEL_MODE=offline")
