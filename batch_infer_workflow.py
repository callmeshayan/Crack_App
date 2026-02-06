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
INFER_FPS = float(os.getenv("RF_INFER_FPS", "2.0"))        # cloud-limited
SAVE_COOLDOWN_S = float(os.getenv("RF_SAVE_COOLDOWN", "0.5"))
ONLY_CLASS = os.getenv("RF_ONLY_CLASS", "crack").strip()   # "" disables
SHOW_PREVIEW = os.getenv("RF_SHOW_PREVIEW", "1").strip() != "0"

# Size thresholds in pixel^2 (tune after you see real values)
SMALL_MAX = float(os.getenv("RF_SMALL_MAX_PX2", "6000"))
MEDIUM_MAX = float(os.getenv("RF_MEDIUM_MAX_PX2", "20000"))

# Save ONLY found
OUT_BASE = Path("data/realtime_results")
FOUND_DIR = OUT_BASE / "found"
REALTIME_FOUND_DIR = OUT_BASE / "realtime_found"
for d in (FOUND_DIR, REALTIME_FOUND_DIR):
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

def pred_conf(p: Dict[str, Any]) -> float:
    return float(p.get("confidence", p.get("score", 0.0)) or 0.0)

def pred_class(p: Dict[str, Any]) -> str:
    return str(p.get("class", p.get("class_name", p.get("label", ""))) or "")

def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """
    Workflow outputs may be dict or list; return first predictions list found.
    """
    if isinstance(result, list):
        for item in result:
            preds = extract_predictions(item)
            if preds:
                return preds
        return []
    if isinstance(result, dict):
        preds = result.get("predictions")
        if isinstance(preds, list):
            return preds
        for v in result.values():
            preds = extract_predictions(v)
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
    """
    Supports:
      - x1,y1,x2,y2
      - x,y,width,height
    """
    if all(k in p for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"])
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if x2i < x1i:
            x1i, x2i = x2i, x1i
        if y2i < y1i:
            y1i, y2i = y2i, y1i
        return x1i, y1i, x2i, y2i

    if all(k in p for k in ("x", "y", "width", "height")):
        x, y = float(p["x"]), float(p["y"])
        w, h = float(p["width"]), float(p["height"])
        x1 = int(round(x - w / 2.0))
        y1 = int(round(y - h / 2.0))
        x2 = int(round(x + w / 2.0))
        y2 = int(round(y + h / 2.0))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    return None

def polygon_points(p: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Supports:
      - "points": [[x,y], ...]
    """
    pts = p.get("points")
    if isinstance(pts, list) and pts and isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 2:
        arr = np.array([(int(round(x)), int(round(y))) for x, y in pts], dtype=np.int32)
        if arr.shape[0] >= 3:
            return arr
    return None

def pred_area_px2(p: Dict[str, Any]) -> float:
    """
    Prefer polygon area if present; else bbox area if present; else 0.
    """
    pts = polygon_points(p)
    if pts is not None:
        return float(abs(cv2.contourArea(pts)))
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
    # OpenCV uses BGR
    if bucket == "S":
        return (0, 255, 0)       # green
    if bucket == "M":
        return (0, 255, 255)     # yellow
    return (0, 0, 255)           # red

def draw_markings(img_bgr: np.ndarray, preds: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw:
      - polygon outline (preferred) or bbox
      - color by size: green small, yellow medium, red large
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    for p in preds:
        area = pred_area_px2(p)
        bucket = size_bucket(area)
        color = size_color_bgr(bucket)

        cls = pred_class(p) or "crack"
        conf = pred_conf(p)

        pts = polygon_points(p)
        if pts is not None:
            # clamp points into image bounds
            pts2 = pts.copy()
            pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
            pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)

            cv2.polylines(out, [pts2], isClosed=True, color=color, thickness=2)

            # label near first point
            x0, y0 = int(pts2[0, 0]), int(pts2[0, 1])
            label = f"{cls} {conf:.2f} {bucket}"
            cv2.putText(out, label, (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            continue

        box = bbox_xyxy(p)
        if box is not None:
            x1, y1, x2, y2 = box
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f} {bucket}"
            cv2.putText(out, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return out

def largest_crack_area(preds: List[Dict[str, Any]]) -> float:
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
    last_saved_found = 0.0

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

            # save found with cooldown
            t = time.time()
            if (t - last_saved_found) < SAVE_COOLDOWN_S:
                continue
            last_saved_found = t

            ts = f"{stamp()}_{int(t*1000)}"
            area_frame = largest_crack_area(preds)
            bucket_frame = size_bucket(area_frame)
            name = f"{bucket_frame}_{int(area_frame)}px2_{ts}"

            annotated = draw_markings(frame, preds)

            for d in (FOUND_DIR, REALTIME_FOUND_DIR):
                cv2.imwrite(str(d / f"{name}.jpg"), annotated)
                (d / f"{name}.json").write_text(json.dumps(result, indent=2))

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

            overlay = f"{st} | dets={c} | best={b:.2f} | MARKER={'ON' if m else 'OFF'}"
            cv2.putText(frame, overlay, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("realtime", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()