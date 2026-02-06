import json
import os
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)



def draw_boxes(img, detections):
    """
    Tries to draw boxes from common Roboflow/Workflow detection schemas.
    Supports:
      - bbox as {x, y, width, height} (center-based) OR {x1,y1,x2,y2}
    """
    for d in detections:
        conf = float(d.get("confidence", d.get("score", 0.0)))
        cls = d.get("class", d.get("class_name", d.get("label", "crack")))

        # Try common bbox formats
        if "x1" in d and "y1" in d and "x2" in d and "y2" in d:
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        else:
            # center-x/center-y with width/height
            x = float(d.get("x", 0))
            y = float(d.get("y", 0))
            w = float(d.get("width", 0))
            h = float(d.get("height", 0))
            x1 = int(round(x - w / 2))
            y1 = int(round(y - h / 2))
            x2 = int(round(x + w / 2))
            y2 = int(round(y + h / 2))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return img


def extract_detections(result):
    """
    Handles both:
      - dict outputs (common)
      - list outputs (workflow can return list of step outputs)
    Returns a list of detection dicts.
    """
    # If workflow returns a list (multiple step outputs), search inside it
    if isinstance(result, list):
        for item in result:
            dets = extract_detections(item)
            if dets:
                return dets
        return []

    # If it's not a dict, nothing to extract
    if not isinstance(result, dict):
        return []

    # Direct common case
    if isinstance(result.get("predictions"), list):
        return result["predictions"]

    # Recursive search for first "detections-like" list
    def looks_like_det(obj):
        return isinstance(obj, dict) and (
            ("confidence" in obj) or ("score" in obj)
        ) and (
            ("x" in obj and "y" in obj) or
            ("x1" in obj and "y1" in obj and "x2" in obj and "y2" in obj)
        )

    def walk(node):
        if isinstance(node, list):
            if node and all(looks_like_det(x) for x in node):
                return node
            for x in node:
                out = walk(x)
                if out is not None:
                    return out
        elif isinstance(node, dict):
            for v in node.values():
                out = walk(v)
                if out is not None:
                    return out
        return None

    found = walk(result)
    return found if found is not None else []

    """
    Workflow outputs differ depending on the blocks used.
    This function searches for the first list of detections-like objects.
    """
    # Most direct case
    if isinstance(result.get("predictions"), list):
        return result["predictions"]

    # Workflow results often have nested steps/outputs
    # Search recursively for a list of dicts that looks like detections
    def looks_like_det(obj):
        return isinstance(obj, dict) and (
            "confidence" in obj or "score" in obj
        ) and (
            "x" in obj or ("x1" in obj and "x2" in obj)
        )

    def walk(node):
        if isinstance(node, list):
            if node and all(looks_like_det(x) for x in node):
                return node
            for x in node:
                out = walk(x)
                if out is not None:
                    return out
        elif isinstance(node, dict):
            for v in node.values():
                out = walk(v)
                if out is not None:
                    return out
        return None

    found = walk(result)
    return found if found is not None else []


def main():
    if len(sys.argv) < 2:
        print("Usage: python infer_image_workflow.py /path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)
    
    api_url = "https://serverless.roboflow.com"
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "").strip()
    workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID", "").strip()

    if not all([api_url, api_key, workspace, workflow_id]):
        raise ValueError("Missing env vars. Check .env for ROBOFLOW_API_URL/KEY/WORKSPACE/WORKFLOW_ID")

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    # Run workflow on image path
    result = client.run_workflow(
        workspace_name=workspace,
        workflow_id=workflow_id,
        images={"image": str(image_path)},
        use_cache=True
    )

    # Save raw JSON
    out_json = image_path.with_suffix(".workflow.json")
    out_json.write_text(json.dumps(result, indent=2))

    # Draw detections
    img = cv2.imread(str(image_path))
    if img is None:
        print("Failed to read image with OpenCV; JSON was still saved.")
        print(f"Saved: {out_json}")
        sys.exit(0)

    dets = extract_detections(result)
    vis = draw_boxes(img.copy(), dets)

    out_img = image_path.with_name(image_path.stem + "_pred.jpg")
    cv2.imwrite(str(out_img), vis)

    print(f"Detections found: {len(dets)}")
    print(f"Saved: {out_img}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
