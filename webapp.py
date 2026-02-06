"""
Flask Web Application for viewing crack detection reports
"""
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from flask import Flask, render_template, send_from_directory, jsonify

app = Flask(__name__)

# Paths
DATA_DIR = Path("data/realtime_results")
SMALL_DIR = DATA_DIR / "small_cracks"
MEDIUM_DIR = DATA_DIR / "medium_cracks"
LARGE_DIR = DATA_DIR / "large_cracks"

def get_detections(directory: Path):
    """Get all detections from a directory."""
    detections = []
    
    if not directory.exists():
        return detections
    
    # Find all JSON files
    for json_file in sorted(directory.glob("*.json"), reverse=True):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get corresponding images
            base_name = json_file.stem
            marked_img = directory / f"{base_name}_marked.jpg"
            original_img = directory / f"{base_name}_original.jpg"
            
            # Parse filename for metadata
            parts = base_name.split('_')
            size_bucket = parts[0] if parts else "?"
            area_px2 = parts[1] if len(parts) > 1 else "?"
            timestamp = parts[2] if len(parts) > 2 else ""
            
            # Get file modification time
            file_time = datetime.fromtimestamp(json_file.stat().st_mtime)
            
            detections.append({
                'id': base_name,
                'size_bucket': size_bucket,
                'area_px2': area_px2.replace('px2', ''),
                'timestamp': file_time.strftime("%Y-%m-%d %H:%M:%S"),
                'marked_image': marked_img.name if marked_img.exists() else None,
                'original_image': original_img.name if original_img.exists() else None,
                'json_file': json_file.name,
                'data': data
            })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return detections

def get_statistics():
    """Get overall statistics."""
    stats = {
        'small': len(list(SMALL_DIR.glob("*.json"))) if SMALL_DIR.exists() else 0,
        'medium': len(list(MEDIUM_DIR.glob("*.json"))) if MEDIUM_DIR.exists() else 0,
        'large': len(list(LARGE_DIR.glob("*.json"))) if LARGE_DIR.exists() else 0,
    }
    stats['total'] = stats['small'] + stats['medium'] + stats['large']
    return stats

@app.route('/')
def index():
    """Dashboard page."""
    stats = get_statistics()
    recent_detections = []
    
    # Get 5 most recent from each category
    for directory in [LARGE_DIR, MEDIUM_DIR, SMALL_DIR]:
        recent_detections.extend(get_detections(directory)[:5])
    
    # Sort by timestamp
    recent_detections.sort(key=lambda x: x['timestamp'], reverse=True)
    recent_detections = recent_detections[:10]  # Top 10 overall
    
    return render_template('index.html', stats=stats, recent=recent_detections)

@app.route('/category/<category>')
def category(category):
    """View all detections in a category."""
    category_map = {
        'small': ('Small Cracks', SMALL_DIR, 'success'),
        'medium': ('Medium Cracks', MEDIUM_DIR, 'warning'),
        'large': ('Large Cracks', LARGE_DIR, 'danger')
    }
    
    if category not in category_map:
        return "Invalid category", 404
    
    title, directory, badge_class = category_map[category]
    detections = get_detections(directory)
    
    return render_template('category.html', 
                          title=title, 
                          category=category,
                          badge_class=badge_class,
                          detections=detections)

@app.route('/detection/<category>/<detection_id>')
def detection_detail(category, detection_id):
    """View detailed information about a specific detection."""
    category_map = {
        'small': SMALL_DIR,
        'medium': MEDIUM_DIR,
        'large': LARGE_DIR
    }
    
    if category not in category_map:
        return "Invalid category", 404
    
    directory = category_map[category]
    json_file = directory / f"{detection_id}.json"
    
    if not json_file.exists():
        return "Detection not found", 404
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    marked_img = directory / f"{detection_id}_marked.jpg"
    original_img = directory / f"{detection_id}_original.jpg"
    
    detection = {
        'id': detection_id,
        'category': category,
        'marked_image': marked_img.name if marked_img.exists() else None,
        'original_image': original_img.name if original_img.exists() else None,
        'data': data
    }
    
    return render_template('detail.html', detection=detection)

@app.route('/images/<category>/<filename>')
def serve_image(category, filename):
    """Serve images from category directories."""
    category_map = {
        'small': SMALL_DIR,
        'medium': MEDIUM_DIR,
        'large': LARGE_DIR
    }
    
    if category not in category_map:
        return "Invalid category", 404
    
    directory = category_map[category]
    return send_from_directory(directory, filename)

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics."""
    return jsonify(get_statistics())

if __name__ == '__main__':
    # Create directories if they don't exist
    for d in [SMALL_DIR, MEDIUM_DIR, LARGE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("\n🚀 Starting Crack Detection Report Web App")
    print("📊 Dashboard: http://localhost:5001")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
