#!/usr/bin/env python3
"""
Batch process all images in a folder using the Roboflow workflow.
Saves annotated images with bounding boxes and JSON results.
"""
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

SMALL_MAX = float(os.getenv("RF_SMALL_MAX_PX2", "6000"))
MEDIUM_MAX = float(os.getenv("RF_MEDIUM_MAX_PX2", "20000"))

def draw_boxes(img, detections):
    """Draw bounding boxes and labels on the image."""
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


def bbox_xyxy(p: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    if all(k in p for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    if all(k in p for k in ("x", "y", "width", "height")):
        x = float(p.get("x", 0))
        y = float(p.get("y", 0))
        w = float(p.get("width", 0))
        h = float(p.get("height", 0))
        x1 = int(round(x - w / 2))
        y1 = int(round(y - h / 2))
        x2 = int(round(x + w / 2))
        y2 = int(round(y + h / 2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    return None


def pred_area_px2(p: Dict[str, Any]) -> float:
    box = bbox_xyxy(p)
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def size_bucket(area_px2: float) -> str:
    if area_px2 <= SMALL_MAX:
        return "S"
    if area_px2 <= MEDIUM_MAX:
        return "M"
    return "L"


def extract_detections(result):
    """Extract detections from workflow result."""
    if isinstance(result, list):
        for item in result:
            dets = extract_detections(item)
            if dets:
                return dets
        return []

    if not isinstance(result, dict):
        return []

    if isinstance(result.get("predictions"), list):
        return result["predictions"]

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


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_process_images.py /path/to/image/folder")
        sys.exit(1)

    folder_path = Path(sys.argv[1])
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Folder not found: {folder_path}")
        sys.exit(1)

    # Get credentials
    api_url = "https://serverless.roboflow.com"
    api_key = os.getenv("RF_API_KEY", "").strip()
    workspace = os.getenv("RF_WORKSPACE", "").strip()
    workflow_id = os.getenv("RF_WORKFLOW_ID", "").strip()

    if not all([api_key, workspace, workflow_id]):
        raise ValueError("Missing env vars. Check .env for RF_API_KEY, RF_WORKSPACE, RF_WORKFLOW_ID")

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    # Create output folders
    output_folder = folder_path.parent / f"{folder_path.name}_results"
    output_folder.mkdir(exist_ok=True)
    
    detected_folder = output_folder / "detected"
    no_cracks_folder = output_folder / "no_cracks"
    detected_folder.mkdir(exist_ok=True)
    no_cracks_folder.mkdir(exist_ok=True)

    # Web dashboard category folders (used by webapp.py)
    dashboard_base = Path("data/realtime_results")
    small_dir = dashboard_base / "small_cracks"
    medium_dir = dashboard_base / "medium_cracks"
    large_dir = dashboard_base / "large_cracks"
    for d in (small_dir, medium_dir, large_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted(
        [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    )

    if not image_files:
        print(f"No images found in {folder_path}")
        sys.exit(0)

    print(f"\n🔍 Processing {len(image_files)} images from {folder_path}")
    print(f"📁 Saving results to {output_folder}\n")

    processed = 0
    detected = 0
    small_count = 0
    medium_count = 0
    large_count = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {image_path.name}...", end=" ")
        
        try:
            # Run workflow on image
            result = client.run_workflow(
                workspace_name=workspace,
                workflow_id=workflow_id,
                images={"image": str(image_path)},
                use_cache=True
            )

            # Extract detections
            dets = extract_detections(result)
            
            # Read and annotate image
            img = cv2.imread(str(image_path))
            if img is None:
                print("❌ Failed to read image")
                continue

            vis = draw_boxes(img.copy(), dets)

            # Save to appropriate folder based on detection
            if len(dets) > 0:
                save_folder = detected_folder
            else:
                save_folder = no_cracks_folder

            # Save results
            out_img = save_folder / f"{image_path.stem}_annotated{image_path.suffix}"
            out_json = save_folder / f"{image_path.stem}.json"
            
            cv2.imwrite(str(out_img), vis)
            out_json.write_text(json.dumps(result, indent=2))

            # Also save to dashboard-compatible folders when crack is detected
            if len(dets) > 0:
                largest_area = max((pred_area_px2(d) for d in dets), default=0.0)
                bucket = size_bucket(largest_area)
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                base_name = f"{bucket}_{int(largest_area)}px2_{ts}_{image_path.stem}"

                if bucket == "S":
                    category_dir = small_dir
                    small_count += 1
                elif bucket == "M":
                    category_dir = medium_dir
                    medium_count += 1
                else:
                    category_dir = large_dir
                    large_count += 1

                marked_path = category_dir / f"{base_name}_marked.jpg"
                original_path = category_dir / f"{base_name}_original.jpg"
                json_path = category_dir / f"{base_name}.json"

                cv2.imwrite(str(marked_path), vis)
                cv2.imwrite(str(original_path), img)
                json_path.write_text(json.dumps(result, indent=2))

            processed += 1
            if len(dets) > 0:
                detected += 1
                print(f"✅ {len(dets)} crack(s) detected")
            else:
                print("⚪ No cracks detected")

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue

    print(f"\n📊 Summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Processed: {processed}")
    print(f"   With cracks: {detected}")
    print(f"   Small / Medium / Large: {small_count} / {medium_count} / {large_count}")
    print(f"   Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
