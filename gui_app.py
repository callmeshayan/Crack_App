import os
import time
import json
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QDoubleSpinBox, QFormLayout, QMessageBox
)

# -------- ENV --------
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
API_KEY = os.getenv("RF_API_KEY", "").strip()
WORKSPACE = os.getenv("RF_WORKSPACE", "").strip()
WORKFLOW_ID = os.getenv("RF_WORKFLOW_ID", "").strip()
if not API_KEY or not WORKSPACE or not WORKFLOW_ID:
    raise ValueError("Missing .env vars: RF_API_KEY, RF_WORKSPACE, RF_WORKFLOW_ID")

client = InferenceHTTPClient("https://serverless.roboflow.com", API_KEY)

OUT_BASE = Path("data/realtime_results")
FOUND_DIR = OUT_BASE / "found"
NOT_FOUND_DIR = OUT_BASE / "not_found"
REALTIME_FOUND_DIR = OUT_BASE / "realtime_found"
for d in (FOUND_DIR, NOT_FOUND_DIR, REALTIME_FOUND_DIR):
    d.mkdir(parents=True, exist_ok=True)

ONLY_CLASS = "crack"

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

def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crack Realtime Controller")

        # state
        self.cap = None
        self.running = False
        self.stop_flag = False

        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()

        self.latest_status = ("idle", 0, 0.0)  # status, dets, best
        self.latest_status_lock = threading.Lock()

        # controls
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(os.getenv("RF_CONF", "0.5")))

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 10.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setValue(2.0)

        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.0, 10.0)
        self.cooldown_spin.setSingleStep(0.1)
        self.cooldown_spin.setValue(0.5)

        # preview + status
        self.preview = QLabel("Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(960, 540)

        self.status_lbl = QLabel("idle | dets=0 | best=0.00")

        # layout
        form = QFormLayout()
        form.addRow("Confidence threshold", self.conf_spin)
        form.addRow("Inference FPS", self.fps_spin)
        form.addRow("Save cooldown (s)", self.cooldown_spin)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)

        root = QVBoxLayout()
        root.addLayout(form)
        root.addLayout(btns)
        root.addWidget(self.status_lbl)
        root.addWidget(self.preview)
        self.setLayout(root)

        # signals
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        # UI timer to refresh preview
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(33)  # ~30 FPS UI refresh

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera", "Camera not available. Check macOS Camera permissions.")
            return

        self.stop_flag = False
        self.running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.infer_thread = threading.Thread(target=self.infer_loop, daemon=True)
        self.capture_thread.start()
        self.infer_thread.start()

    def stop(self):
        if not self.running:
            return
        self.stop_flag = True
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def capture_loop(self):
        while not self.stop_flag and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.latest_frame_lock:
                self.latest_frame = frame

    def infer_loop(self):
        last_infer = 0.0
        last_saved_found = 0.0
        last_saved_nf = 0.0

        while not self.stop_flag:
            infer_fps = float(self.fps_spin.value())
            min_interval = 1.0 / max(infer_fps, 0.1)

            now = time.time()
            if now - last_infer < min_interval:
                time.sleep(0.01)
                continue

            with self.latest_frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
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

                preds = extract_predictions(result)

                conf_thr = float(self.conf_spin.value())
                preds = [
                    p for p in preds
                    if pred_conf(p) >= conf_thr and (not ONLY_CLASS or pred_class(p).lower() == ONLY_CLASS.lower())
                ]

                best = max((pred_conf(p) for p in preds), default=0.0)
                count = len(preds)
                found = count > 0

                with self.latest_status_lock:
                    self.latest_status = ("found" if found else "not_found", count, best)

                cooldown = float(self.cooldown_spin.value())
                t = time.time()
                name = f"{stamp()}_{int(t*1000)}"

                if found and (t - last_saved_found) >= cooldown:
                    last_saved_found = t
                    for d in (FOUND_DIR, REALTIME_FOUND_DIR):
                        base = d / name
                        cv2.imwrite(str(base) + ".jpg", frame)
                        Path(str(base) + ".json").write_text(json.dumps(result, indent=2))

                if (not found) and (t - last_saved_nf) >= cooldown:
                    last_saved_nf = t
                    base = NOT_FOUND_DIR / name
                    cv2.imwrite(str(base) + ".jpg", frame)
                    Path(str(base) + ".json").write_text(json.dumps(result, indent=2))

            except Exception:
                with self.latest_status_lock:
                    self.latest_status = ("error", 0, 0.0)
                time.sleep(0.05)

    def refresh_ui(self):
        # update status
        with self.latest_status_lock:
            st, count, best = self.latest_status
        self.status_lbl.setText(f"{st} | dets={count} | best={best:.2f}")

        # update preview
        with self.latest_frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.preview.width(), self.preview.height(), Qt.KeepAspectRatio)
        self.preview.setPixmap(pix)

    def closeEvent(self, event):
        self.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    w = App()
    w.show()
    app.exec()