import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Detection:
    cls: str
    confidence: float
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


class RoboflowClient:
    def __init__(self):
        api_url = "https://serverless.roboflow.com"
        self.api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
        self.base_url = os.getenv("ROBOFLOW_URL", "").strip().rstrip("/")
        self.conf = float(os.getenv("ROBOFLOW_CONF", "0.25"))

        if not self.api_key or not self.base_url:
            raise ValueError(
                "Missing ROBOFLOW_API_KEY or ROBOFLOW_URL in .env"
            )

        # Final endpoint
        self.url = f"{self.base_url}?api_key={self.api_key}"

    @staticmethod
    def _b64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        payload = {"image": self._b64(image_bytes)}
        r = requests.post(self.url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def parse_detections(
        self,
        resp: Dict[str, Any],
        only_class: Optional[str] = "crack",
    ) -> List[Detection]:
        preds = resp.get("predictions", []) or []
        out: List[Detection] = []

        for p in preds:
            cls = str(p.get("class", ""))
            conf = float(p.get("confidence", 0.0))
            if conf < self.conf:
                continue
            if only_class and cls.lower() != only_class.lower():
                continue

            # Roboflow commonly returns center x/y and width/height in pixels
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            w = float(p.get("width", 0.0))
            h = float(p.get("height", 0.0))

            x1 = int(round(x - w / 2))
            y1 = int(round(y - h / 2))
            x2 = int(round(x + w / 2))
            y2 = int(round(y + h / 2))

            out.append(Detection(cls=cls, confidence=conf, box=(x1, y1, x2, y2)))

        return out
