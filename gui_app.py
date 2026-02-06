import os
import time
import json
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# ---- optional but REQUIRED for rle_mask decoding ----
try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCOTOOLS = True
except Exception:
    HAS_PYCOCOTOOLS = False


# ================== ENV ==================
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

API_KEY = os.getenv("RF_API_KEY", "").strip()
WORKSPACE = os.getenv("RF_WORKSPACE", "").strip()
WORKFLOW_ID = os.getenv("RF_WORKFLOW_ID", "").strip()

if not API_KEY or not WORKSPACE or not WORKFLOW_ID:
    raise ValueError("Missing .env vars: RF_API_KEY, RF_WORKSPACE, RF_WORKFLOW_ID")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)

# ================== SETTINGS ==================
CONF_THRESH = float(os.getenv("RF_CONF", "0.5"))
INFER_FPS = float(os.getenv("RF_INFER_FPS", "2.0"))
SAVE_COOLDOWN_S = float(os.getenv("RF_SAVE_COOLDOWN", "0.5"))
ONLY_CLASS = os.getenv("RF_ONLY_CLASS", "crack").strip()  # "" disables
SHOW_PREVIEW = os.getenv("RF_SHOW_PREVIEW", "1").strip() != "0"

# size thresholds in px^2
SMALL_MAX = float(os.getenv("RF_SMALL_MAX_PX2", "6000"))
MEDIUM_MAX = float(os.getenv("RF_MEDIUM_MAX_PX2", "20000"))

# save ONLY found
OUT_BASE = Path("data/realtime_results")
FOUND_DIR = OUT_BASE / "found"
NOT_FOUND_DIR = OUT_BASE / "not_found"
REALTIME_FOUND_DIR = OUT_BASE / "realtime_found"

# Size-based categorization directories
SMALL_DIR = OUT_BASE / "small_cracks"
MEDIUM_DIR = OUT_BASE / "medium_cracks"
LARGE_DIR = OUT_BASE / "large_cracks"

for d in (FOUND_DIR, NOT_FOUND_DIR, REALTIME_FOUND_DIR, SMALL_DIR, MEDIUM_DIR, LARGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ================== MARKER (1s pulse boolean) ==================
marker_active = False
marker_until = 0.0
marker_lock = threading.Lock()

def trigger_marker_pulse(duration_s: float = 1.0):
    global marker_active, marker_until
    now = time.time()
    with marker_lock:
        if not marker_active:
            marker_active = True
            marker_until = now + duration_s

def update_marker_state():
    global marker_active
    now = time.time()
    with marker_lock:
        if marker_active and now >= marker_until:
            marker_active = False

# ================== HELPERS ==================
def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def get_bbox_from_pred(p: Dict[str, Any]) -> tuple:
    """Extract bounding box coordinates from prediction."""
    x = float(p.get("x", 0.0))
    y = float(p.get("y", 0.0))
    w = float(p.get("width", 0.0))
    h = float(p.get("height", 0.0))
    
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    
    return (x1, y1, x2, y2, w, h)

def draw_predictions_on_frame(frame, preds: List[Dict[str, Any]]) -> Any:
    """Draw bounding boxes and labels on frame."""
    marked_frame = frame.copy()
    
    for p in preds:
        x1, y1, x2, y2, w, h = get_bbox_from_pred(p)
        conf = pred_conf(p)
        cls = pred_class(p)
        area = w * h
        
        # Determine color based on size
        if area < SMALL_MAX:
            color = (0, 255, 255)  # Yellow for small
            size_label = "SMALL"
        elif area < MEDIUM_MAX:
            color = (0, 165, 255)  # Orange for medium
            size_label = "MEDIUM"
        else:
            color = (0, 0, 255)  # Red for large
            size_label = "LARGE"
        
        # Draw bounding box
        cv2.rectangle(marked_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence and size
        label = f"{cls} {conf:.2f} [{size_label}]"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            marked_frame,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            marked_frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return marked_frame

def categorize_by_size(preds: List[Dict[str, Any]]) -> str:
    """Determine size category based on largest crack."""
    if not preds:
        return "none"
    
    max_area = 0
    for p in preds:
        _, _, _, _, w, h = get_bbox_from_pred(p)
        area = w * h
        max_area = max(max_area, area)
    
    if max_area < SMALL_MAX:
        return "small"
    elif max_area < MEDIUM_MAX:
        return "medium"
    else:
        return "large"

def pred_conf(p: Dict[str, Any]) -> float:
    return float(p.get("confidence", p.get("score", 0.0)) or 0.0)

def pred_class(p: Dict[str, Any]) -> str:
    return str(p.get("class", p.get("class_name", p.get("label", ""))) or "")

def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """
    Your workflow returns a list with one dict:
      { "predictions": { "image":..., "predictions":[...] }, "visualization": ... }
    This function returns the inner list under any nested 'predictions'.
    """
    if isinstance(result, list):
        for item in result:
            preds = extract_predictions(item)
            if preds:
                return preds
        return []

    if isinstance(result, dict):
        v = result.get("predictions")
        if isinstance(v, list):
            return v
        if isinstance(v, dict) and isinstance(v.get("predictions"), list):
            return v["predictions"]

        for vv in result.values():
            preds = extract_predictions(vv)
            if preds:
                return preds

    return []

def filter_preds(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in preds:
        if pred_conf(p) < CONF_THRESH:
            continue
        if ONLY_CLASS and pred_class(p).lower() != ONLY_CLASS.lower():
            continue
        out.append(p)
    return out

def bbox_xyxy(p: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    if all(k in p for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return x1, y1, x2, y2
    if all(k in p for k in ("x", "y", "width", "height")):
        x, y = float(p["x"]), float(p["y"])
        w, h = float(p["width"]), float(p["height"])
        x1 = int(round(x - w / 2))
        y1 = int(round(y - h / 2))
        x2 = int(round(x + w / 2))
        y2 = int(round(y + h / 2))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return x1, y1, x2, y2
    return None

def decode_rle_mask_to_binary(p: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Returns binary mask (H,W) uint8 0/255 if rle_mask exists and pycocotools is installed.
    """
    if not HAS_PYCOCOTOOLS:
        return None
    rm = p.get("rle_mask")
    if not isinstance(rm, dict) or "size" not in rm or "counts" not in rm:
        return None
    try:
        m = mask_utils.decode(rm)  # shape (H,W,1) or (H,W)
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m.astype(np.uint8) * 255)
    except Exception:
        return None

def mask_area_px2(mask_u8: np.ndarray) -> float:
    return float(cv2.countNonZero(mask_u8))

def pred_area_px2(p: Dict[str, Any]) -> float:
    m = decode_rle_mask_to_binary(p)
    if m is not None:
        return mask_area_px2(m)
    box = bbox_xyxy(p)
    if box is not None:
        x1, y1, x2, y2 = box
        return float(max(0, x2 - x1) * max(0, y2 - y1))
    return 0.0

def size_bucket(area_px2: float) -> str:
    if area_px2 <= SMALL_MAX:
        return "S"
    if area_px2 <= MEDIUM_MAX:
        return "M"
    return "L"

def size_color_bgr(bucket: str) -> Tuple[int, int, int]:
    if bucket == "S":
        return (0, 255, 0)       # green
    if bucket == "M":
        return (0, 255, 255)     # yellow
    return (0, 0, 255)           # red

def draw_markings(img_bgr: np.ndarray, preds: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw segmentation outlines when rle_mask exists; otherwise draw bbox.
    Color is based on crack size bucket.
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    for p in preds:
        conf = pred_conf(p)
        cls = pred_class(p) or "crack"

        area = pred_area_px2(p)
        bucket = size_bucket(area)
        color = size_color_bgr(bucket)

        mask_u8 = decode_rle_mask_to_binary(p)
        if mask_u8 is not None:
            # find contours on mask and draw outline
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if len(c) >= 3:
                    cv2.polylines(out, [c], isClosed=True, color=color, thickness=2)

            # label near largest contour bbox
            if contours:
                cmax = max(contours, key=cv2.contourArea)
                x, y, ww, hh = cv2.boundingRect(cmax)
                label = f"{cls} {conf:.2f} {bucket}"
                cv2.putText(out, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            continue

        # fallback: bbox
        box = bbox_xyxy(p)
        if box is not None:
            x1, y1, x2, y2 = box
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f} {bucket}"
            cv2.putText(out, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return out

def largest_area(preds: List[Dict[str, Any]]) -> float:
    return max((pred_area_px2(p) for p in preds), default=0.0)

# ================== SHARED STATE ==================
latest_frame = None
latest_frame_lock = threading.Lock()

latest_status = {"status": "idle", "count": 0, "best": 0.0}
latest_status_lock = threading.Lock()

stop_flag = False

# ================== INFERENCE THREAD ==================
def inference_loop():
    global stop_flag

    min_interval = 1.0 / max(INFER_FPS, 0.1)
    last_infer = 0.0
    last_saved = 0.0

    while not stop_flag:
        now = time.time()
        if now - last_infer < min_interval:
            time.sleep(0.01)
            continue

        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            continue

        last_infer = now

        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
                f.write(jpg.tobytes())
                f.flush()
                result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW_ID,
                    images={"image": f.name},
                    use_cache=True
                )

            preds = filter_preds(extract_predictions(result))
            count = len(preds)
            best = max((pred_conf(p) for p in preds), default=0.0)
            found = count > 0

            if found:
                trigger_marker_pulse(1.0)
            update_marker_state()

            with latest_status_lock:
                latest_status["status"] = "found" if found else "not_found"
                latest_status["count"] = count
                latest_status["best"] = best

            # discard not found
            if not found:
                continue

            # cooldown for saving
            t = time.time()
            if (t - last_saved) < SAVE_COOLDOWN_S:
                continue
            last_saved = t

            # name by largest crack size in frame
            area_frame = largest_area(preds)
            bucket_frame = size_bucket(area_frame)
            name = f"{bucket_frame}_{int(area_frame)}px2_{stamp()}_{int(t*1000)}"

            annotated = draw_markings(frame, preds)

            # Save to found directories
            for d in (FOUND_DIR, REALTIME_FOUND_DIR):
                cv2.imwrite(str(d / f"{name}_marked.jpg"), annotated)
                cv2.imwrite(str(d / f"{name}_original.jpg"), frame)
                (d / f"{name}.json").write_text(json.dumps(result, indent=2))
            
            # Save to size-categorized directory
            if bucket_frame == "S":
                size_dir = SMALL_DIR
            elif bucket_frame == "M":
                size_dir = MEDIUM_DIR
            else:
                size_dir = LARGE_DIR
            
            cv2.imwrite(str(size_dir / f"{name}_marked.jpg"), annotated)
            cv2.imwrite(str(size_dir / f"{name}_original.jpg"), frame)
            (size_dir / f"{name}.json").write_text(json.dumps(result, indent=2))

        except Exception:
            with latest_status_lock:
                latest_status["status"] = "error"
                latest_status["count"] = 0
                latest_status["best"] = 0.0
            time.sleep(0.05)

# ================== MAIN ==================
def main():
    global latest_frame, stop_flag

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    worker = threading.Thread(target=inference_loop, daemon=True)
    worker.start()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        with latest_frame_lock:
            latest_frame = frame

        if SHOW_PREVIEW:
            with latest_status_lock:
                st = latest_status["status"]
                c = latest_status["count"]
                b = latest_status["best"]

            with marker_lock:
                m = marker_active

            warn = "" if HAS_PYCOCOTOOLS else " | NO pycocotools (no mask outlines)"
            overlay = f"{st} | dets={c} | best={b:.2f} | MARKER={'ON' if m else 'OFF'}{warn}"
            cv2.putText(frame, overlay, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("realtime", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()