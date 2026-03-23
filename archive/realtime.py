import os
import time
import json
import threading
from pathlib import Path
from typing import Any, Dict, List
import tempfile

import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# ---------- ENV ----------
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

# ---------- SETTINGS ----------
CONF_THRESH = float(os.getenv("RF_CONF", "0.5"))
INFER_FPS = 2.0                 # target inference rate (cloud-limited); 1–3 is typical
SAVE_COOLDOWN_S = 0.5           # minimum time between saved frames per category
ONLY_CLASS = "crack"            # set "" to disable class filtering
SHOW_PREVIEW = True

OUT_BASE = Path("data/realtime_results")
FOUND_DIR = OUT_BASE / "found"
NOT_FOUND_DIR = OUT_BASE / "not_found"
REALTIME_FOUND_DIR = OUT_BASE / "realtime_found"

FOUND_DIR.mkdir(parents=True, exist_ok=True)
NOT_FOUND_DIR.mkdir(parents=True, exist_ok=True)
REALTIME_FOUND_DIR.mkdir(parents=True, exist_ok=True)

# ---------- HELPERS ----------
def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """
    Workflow outputs can be dict or list. Return first 'predictions'-like list found.
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


def pred_conf(p: Dict[str, Any]) -> float:
    return float(p.get("confidence", p.get("score", 0.0)) or 0.0)


def pred_class(p: Dict[str, Any]) -> str:
    return str(p.get("class", p.get("class_name", p.get("label", ""))) or "")


def filter_preds(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in preds:
        if pred_conf(p) < CONF_THRESH:
            continue
        if ONLY_CLASS:
            if pred_class(p).lower() != ONLY_CLASS.lower():
                continue
        out.append(p)
    return out


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ---------- SHARED STATE (capture -> inference) ----------
latest_frame = None
latest_frame_lock = threading.Lock()

latest_result = {"status": "idle", "best": 0.0, "count": 0, "predictions": []}
latest_result_lock = threading.Lock()

stop_flag = False


# ---------- INFERENCE WORKER ----------
def inference_loop():
    global stop_flag

    min_interval = 1.0 / max(INFER_FPS, 0.1)
    last_infer_time = 0.0
    last_saved_found = 0.0
    last_saved_not_found = 0.0

    while not stop_flag:
        now = time.time()
        if now - last_infer_time < min_interval:
            time.sleep(0.01)
            continue

        # grab the newest frame (drop older ones)
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        last_infer_time = now

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
                    use_cache=True,
                )

            preds = filter_preds(extract_predictions(result))
            best = max((pred_conf(p) for p in preds), default=0.0)
            count = len(preds)
            found = count > 0

            # update overlay status
            with latest_result_lock:
                latest_result["status"] = "found" if found else "not_found"
                latest_result["best"] = best
                latest_result["count"] = count
                latest_result["predictions"] = preds

            # categorize + save (cooldown)
            t = time.time()
            name = f"{stamp()}_{int(t*1000)}"

            if found and (t - last_saved_found) >= SAVE_COOLDOWN_S:
                last_saved_found = t

                # Save in found
                base_found = FOUND_DIR / name
                cv2.imwrite(str(base_found) + ".jpg", frame)
                Path(str(base_found) + ".json").write_text(json.dumps(result, indent=2))

                # ALSO save in realtime_found
                base_rt = REALTIME_FOUND_DIR / name
                cv2.imwrite(str(base_rt) + ".jpg", frame)
                Path(str(base_rt) + ".json").write_text(json.dumps(result, indent=2))

            # Removed: Not saving undetected frames anymore

        except Exception:
            with latest_result_lock:
                latest_result["status"] = "error"
                latest_result["best"] = 0.0
                latest_result["count"] = 0
                latest_result["predictions"] = []
            time.sleep(0.05)


# ---------- MAIN (capture + display) ----------
def main():
    global latest_frame, stop_flag

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS; on Pi use different backend
    if not cap.isOpened():
        raise RuntimeError("Camera not available (macOS camera permissions / close other apps).")

    worker = threading.Thread(target=inference_loop, daemon=True)
    worker.start()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        with latest_frame_lock:
            latest_frame = frame

        if SHOW_PREVIEW:
            with latest_result_lock:
                status = latest_result["status"]
                best = latest_result["best"]
                count = latest_result["count"]
                preds = latest_result["predictions"].copy()

            # Draw bounding boxes
            for pred in preds:
                # Get bounding box coordinates
                x = int(pred.get("x", 0))
                y = int(pred.get("y", 0))
                w = int(pred.get("width", 0))
                h = int(pred.get("height", 0))
                
                # Convert center coordinates to top-left coordinates
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence
                conf = pred_conf(pred)
                label = f"{pred_class(pred)}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            overlay = f"{status} | dets={count} | best={best:.2f} | infer_fps~{INFER_FPS}"
            cv2.putText(frame, overlay, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("realtime", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()