import json
import sys
from pathlib import Path

import cv2

from roboflow_client import RoboflowClient


def draw_detections(img, dets):
    for d in dets:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.cls} {d.confidence:.2f}"
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return img


def main():
    if len(sys.argv) < 2:
        print("Usage: python infer_image.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    client = RoboflowClient()

    img = cv2.imread(str(image_path))
    if img is None:
        print("Failed to read image.")
        sys.exit(1)

    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        print("Failed to encode image.")
        sys.exit(1)

    resp = client.predict_bytes(buf.tobytes())
    dets = client.parse_detections(resp, only_class="crack")

    # Save raw response
    out_json = image_path.with_suffix(".predictions.json")
    out_json.write_text(json.dumps(resp, indent=2))

    # Save annotated image
    out_img = image_path.with_name(image_path.stem + "_pred.jpg")
    vis = draw_detections(img.copy(), dets)
    cv2.imwrite(str(out_img), vis)

    print(f"Cracks found: {len(dets)}")
    print(f"Saved: {out_img}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
