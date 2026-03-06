"""
Raspberry Pi 5 Dual CSI Camera Real-time Crack Detection (libcamera/rpicam)
- Captures frames from CSI cameras using rpicam-vid MJPEG stream (NOT /dev/video0)
- Runs Roboflow workflow inference (InferenceHTTPClient)
- Supports two cameras simultaneously
"""

import os
import time
import json
import threading
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# ---------------- ENV ----------------
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

# ---------------- SETTINGS ----------------
CONF_THRESH = float(os.getenv("RF_CONF", "0.5"))
INFER_FPS = 2.0
SAVE_COOLDOWN_S = 0.5
ONLY_CLASS = "crack"  # "" to disable class filtering

# IMPORTANT: if you're SSH-ing in, keep this False (imshow won't work)
SHOW_PREVIEW = False

ENABLE_PREPROCESSING = True
ENABLE_PERSISTENCE = True
PERSISTENCE_FRAMES = 3
BLUR_THRESHOLD = 100.0
MIN_CRACK_AREA = 100

SEVERITY_CRITICAL = 0.85
SEVERITY_HIGH = 0.70
SEVERITY_MEDIUM = 0.55

# Camera capture settings (rpicam)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_FPS = 30

# Dual CSI camera IDs for rpicam (NOT /dev/videoX)
CAMERA_0_ID = 0
CAMERA_1_ID = 1

OUT_BASE = Path("data/realtime_results")
FOUND_DIR_CAM0 = OUT_BASE / "camera0_found"
FOUND_DIR_CAM1 = OUT_BASE / "camera1_found"
REALTIME_FOUND_DIR_CAM0 = OUT_BASE / "camera0_realtime"
REALTIME_FOUND_DIR_CAM1 = OUT_BASE / "camera1_realtime"

for p in [FOUND_DIR_CAM0, FOUND_DIR_CAM1, REALTIME_FOUND_DIR_CAM0, REALTIME_FOUND_DIR_CAM1]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------- HELPERS ----------------
def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def extract_predictions(result: Any) -> List[Dict[str, Any]]:
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
        if ONLY_CLASS and pred_class(p).lower() != ONLY_CLASS.lower():
            continue
        out.append(p)
    return out


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if not ENABLE_PREPROCESSING:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
    enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(enhanced_bgr, 0.7, frame, 0.3, 0)
    return blended


def check_frame_quality(frame: np.ndarray) -> Tuple[bool, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (lap_var >= BLUR_THRESHOLD), float(lap_var)


def calculate_crack_area(pred: Dict[str, Any]) -> float:
    w = float(pred.get("width", 0))
    h = float(pred.get("height", 0))
    return w * h


def classify_severity(confidence: float) -> str:
    if confidence >= SEVERITY_CRITICAL:
        return "CRITICAL"
    if confidence >= SEVERITY_HIGH:
        return "HIGH"
    if confidence >= SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def draw_detections(frame: np.ndarray, preds: List[Dict[str, Any]]) -> np.ndarray:
    for pred in preds:
        x = int(pred.get("x", 0))
        y = int(pred.get("y", 0))
        w = int(pred.get("width", 0))
        h = int(pred.get("height", 0))

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        conf = pred_conf(pred)
        severity = classify_severity(conf)

        if severity == "CRITICAL":
            color, thickness = (0, 0, 255), 3
        elif severity == "HIGH":
            color, thickness = (0, 165, 255), 3
        elif severity == "MEDIUM":
            color, thickness = (0, 255, 255), 2
        else:
            color, thickness = (0, 255, 0), 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        area = calculate_crack_area(pred)
        label = f"{pred_class(pred)} [{severity}] {conf:.2f} ({int(area)}px)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


# ---------------- STATE ----------------
class CameraState:
    def __init__(self, camera_id: int, found_dir: Path, realtime_dir: Path):
        self.camera_id = camera_id
        self.found_dir = found_dir
        self.realtime_dir = realtime_dir

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.latest_result = {"status": "idle", "best": 0.0, "count": 0, "predictions": [], "blur_score": 0.0}
        self.result_lock = threading.Lock()

        self.detection_history = deque(maxlen=PERSISTENCE_FRAMES)
        self.history_lock = threading.Lock()

        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "skipped_blurry": 0,
            "detections_found": 0,
            "critical_cracks": 0,
            "high_cracks": 0,
            "medium_cracks": 0,
            "low_cracks": 0,
            "total_saved": 0,
        }
        self.stats_lock = threading.Lock()

        self.stop_flag = False

    def update_detection_history(self, has_detection: bool):
        with self.history_lock:
            self.detection_history.append(has_detection)

    def check_detection_persistence(self) -> bool:
        if not ENABLE_PERSISTENCE:
            return True
        with self.history_lock:
            if len(self.detection_history) < PERSISTENCE_FRAMES:
                return False
            return all(self.detection_history)


# ---------------- CAPTURE (rpicam pipe) ----------------
def _spawn_rpicam_mjpeg(camera_id: int) -> subprocess.Popen:
    cmd = [
        "rpicam-vid",
        "--camera", str(camera_id),
        "-n",  # no preview
        "--codec", "mjpeg",
        "--width", str(CAMERA_WIDTH),
        "--height", str(CAMERA_HEIGHT),
        "--framerate", str(CAPTURE_FPS),
        "-t", "0",
        "-o", "-",
    ]
    # stderr suppressed to keep console clean; remove DEVNULL if you want verbose logs
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)


def camera_capture_thread(cam_state: CameraState, rpicam_camera_id: int):
    """
    Reads MJPEG frames from rpicam-vid stdout and decodes into OpenCV frames.
    """
    proc = _spawn_rpicam_mjpeg(rpicam_camera_id)

    data = bytearray()
    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    print(f"[CAM{cam_state.camera_id}] rpicam-vid started (camera={rpicam_camera_id})")

    try:
        while not cam_state.stop_flag:
            chunk = proc.stdout.read(4096)
            if not chunk:
                time.sleep(0.01)
                continue

            data.extend(chunk)

            # find full JPEG
            start = data.find(SOI)
            if start == -1:
                # keep buffer bounded
                if len(data) > 2_000_000:
                    data.clear()
                continue

            end = data.find(EOI, start + 2)
            if end == -1:
                # need more bytes
                continue

            jpg = data[start:end + 2]
            del data[:end + 2]

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            with cam_state.frame_lock:
                cam_state.latest_frame = frame

    finally:
        cam_state.stop_flag = True
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        print(f"[CAM{cam_state.camera_id}] capture stopped")


# ---------------- INFERENCE ----------------
def inference_loop(cam_state: CameraState):
    min_interval = 1.0 / max(INFER_FPS, 0.1)
    last_infer_time = 0.0
    last_saved_found = 0.0

    while not cam_state.stop_flag:
        now = time.time()
        if now - last_infer_time < min_interval:
            time.sleep(0.01)
            continue

        with cam_state.frame_lock:
            frame = None if cam_state.latest_frame is None else cam_state.latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        with cam_state.stats_lock:
            cam_state.stats["total_frames"] += 1

        is_good, blur_score = check_frame_quality(frame)
        if not is_good:
            with cam_state.stats_lock:
                cam_state.stats["skipped_blurry"] += 1
            cam_state.update_detection_history(False)
            with cam_state.result_lock:
                cam_state.latest_result["status"] = "blurry"
                cam_state.latest_result["blur_score"] = blur_score
            time.sleep(0.02)
            continue

        last_infer_time = now
        processed = preprocess_frame(frame)

        ok, jpg = cv2.imencode(".jpg", processed)
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

            with cam_state.stats_lock:
                cam_state.stats["processed_frames"] += 1

            preds = filter_preds(extract_predictions(result))
            preds = [p for p in preds if calculate_crack_area(p) >= MIN_CRACK_AREA]

            best = max((pred_conf(p) for p in preds), default=0.0)
            count = len(preds)
            found = count > 0

            cam_state.update_detection_history(found)

            if found:
                with cam_state.stats_lock:
                    cam_state.stats["detections_found"] += 1
                    for p in preds:
                        sev = classify_severity(pred_conf(p))
                        if sev == "CRITICAL":
                            cam_state.stats["critical_cracks"] += 1
                        elif sev == "HIGH":
                            cam_state.stats["high_cracks"] += 1
                        elif sev == "MEDIUM":
                            cam_state.stats["medium_cracks"] += 1
                        else:
                            cam_state.stats["low_cracks"] += 1

            with cam_state.result_lock:
                cam_state.latest_result["status"] = "found" if found else "not_found"
                cam_state.latest_result["best"] = best
                cam_state.latest_result["count"] = count
                cam_state.latest_result["predictions"] = preds
                cam_state.latest_result["blur_score"] = blur_score

            is_persistent = cam_state.check_detection_persistence()
            t = time.time()
            name = f"cam{cam_state.camera_id}_{stamp()}_{int(t*1000)}"

            if found and is_persistent and (t - last_saved_found) >= SAVE_COOLDOWN_S:
                last_saved_found = t
                with cam_state.stats_lock:
                    cam_state.stats["total_saved"] += 1

                metadata = {
                    "result": result,
                    "blur_score": blur_score,
                    "detections": count,
                    "max_confidence": best,
                    "timestamp": t,
                    "preprocessing_enabled": ENABLE_PREPROCESSING,
                    "persistence_frames": PERSISTENCE_FRAMES,
                }

                base_found = cam_state.found_dir / name
                cv2.imwrite(str(base_found) + ".jpg", frame)
                cv2.imwrite(str(base_found) + "_enhanced.jpg", processed)
                Path(str(base_found) + ".json").write_text(json.dumps(metadata, indent=2))

                base_rt = cam_state.realtime_dir / name
                cv2.imwrite(str(base_rt) + ".jpg", frame)
                cv2.imwrite(str(base_rt) + "_enhanced.jpg", processed)
                Path(str(base_rt) + ".json").write_text(json.dumps(metadata, indent=2))

        except Exception:
            with cam_state.result_lock:
                cam_state.latest_result = {"status": "error", "best": 0.0, "count": 0, "predictions": [], "blur_score": blur_score}
            time.sleep(0.05)


# ---------------- MAIN ----------------
def main():
    print("Starting Pi 5 Dual CSI Camera Crack Detection (rpicam pipeline)")
    print(f"Camera0 rpicam id: {CAMERA_0_ID}")
    print(f"Camera1 rpicam id: {CAMERA_1_ID}")
    print("Press Ctrl+C to stop.")

    cam0 = CameraState(0, FOUND_DIR_CAM0, REALTIME_FOUND_DIR_CAM0)
    cam1 = CameraState(1, FOUND_DIR_CAM1, REALTIME_FOUND_DIR_CAM1)

    t_cap0 = threading.Thread(target=camera_capture_thread, args=(cam0, CAMERA_0_ID), daemon=True)
    t_cap1 = threading.Thread(target=camera_capture_thread, args=(cam1, CAMERA_1_ID), daemon=True)
    t_inf0 = threading.Thread(target=inference_loop, args=(cam0,), daemon=True)
    t_inf1 = threading.Thread(target=inference_loop, args=(cam1,), daemon=True)

    t_cap0.start()
    t_cap1.start()
    t_inf0.start()
    t_inf1.start()

    try:
        while True:
            if SHOW_PREVIEW:
                with cam0.frame_lock:
                    f0 = None if cam0.latest_frame is None else cam0.latest_frame.copy()
                with cam1.frame_lock:
                    f1 = None if cam1.latest_frame is None else cam1.latest_frame.copy()

                frames = []
                if f0 is not None:
                    with cam0.result_lock:
                        preds0 = list(cam0.latest_result.get("predictions", []))
                        st0 = cam0.latest_result.get("status", "")
                        b0 = cam0.latest_result.get("best", 0.0)
                        c0 = cam0.latest_result.get("count", 0)
                        q0 = cam0.latest_result.get("blur_score", 0.0)
                    f0 = draw_detections(f0, preds0)
                    cv2.putText(f0, f"CAM0 | {st0} | dets={c0} | conf={b0:.2f} | blur={q0:.1f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    frames.append(f0)

                if f1 is not None:
                    with cam1.result_lock:
                        preds1 = list(cam1.latest_result.get("predictions", []))
                        st1 = cam1.latest_result.get("status", "")
                        b1 = cam1.latest_result.get("best", 0.0)
                        c1 = cam1.latest_result.get("count", 0)
                        q1 = cam1.latest_result.get("blur_score", 0.0)
                    f1 = draw_detections(f1, preds1)
                    cv2.putText(f1, f"CAM1 | {st1} | dets={c1} | conf={b1:.2f} | blur={q1:.1f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    frames.append(f1)

                if len(frames) == 2:
                    cv2.imshow("Pi5 Dual Camera - Crack Detection", cv2.hconcat(frames))
                elif len(frames) == 1:
                    cv2.imshow("Pi5 Camera - Crack Detection", frames[0])

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        cam0.stop_flag = True
        cam1.stop_flag = True
        time.sleep(0.5)
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

        print("\n================= STATS =================")
        for cid, cs in [(0, cam0), (1, cam1)]:
            with cs.stats_lock:
                s = dict(cs.stats)
            print(f"\nCamera {cid}:")
            print(f"  Total Frames: {s['total_frames']}")
            print(f"  Processed: {s['processed_frames']}")
            print(f"  Skipped (blurry): {s['skipped_blurry']}")
            print(f"  Detections Found: {s['detections_found']}")
            print(f"  Saved: {s['total_saved']}")
            print(f"  Severity: critical={s['critical_cracks']} high={s['high_cracks']} medium={s['medium_cracks']} low={s['low_cracks']}")
        print("\nDone.")


if __name__ == "__main__":
    main()
