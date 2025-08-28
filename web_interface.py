#!/usr/bin/env python3
"""
Cucumber HTP Web Interface
Upload images and test your trained model in real-time
"""

import os
import sys
import base64
import io
import json
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from infer_seg import CucumberPhenotypingPipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and pipeline
pipeline = None
model_loaded = False

def load_model():
    """Load the trained model"""
    global pipeline, model_loaded
    try:
        model_path = "outputs/models/train/weights/best.pt"
        if not os.path.exists(model_path):
            return False, "Model file not found. Please ensure training is complete."

        print("🔧 Initializing pipeline components...")
        pipeline = CucumberPhenotypingPipeline(
            model_path=model_path,
            confidence_threshold=0.5,
            iou_threshold=0.45
        )
        print(f"✅ YOLO model loaded: {model_path}")
        print("✅ Pipeline initialization complete!")
        model_loaded = True
        return True, "Model loaded successfully!"
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False
        return False, f"Error loading model: {str(e)}"

def process_image_file(temp_path: str):
    """Process image through the pipeline using a temporary file path"""
    try:
        if pipeline is None:
            return False, "Model not loaded"

        # Load image for processing
        img = cv2.imread(temp_path)
        if img is None:
            return False, "Failed to load image"
        
        img_height, img_width = img.shape[:2]
        print(f"📐 Image dimensions: {img_width}x{img_height}")
        
        # Get YOLO detection results directly
        detection_results = pipeline.model.predict(
            source=img, 
            imgsz=1024, 
            conf=pipeline.confidence_threshold, 
            iou=pipeline.iou_threshold, 
            verbose=False
        )[0]
        
        # Extract masks and boxes
        masks = detection_results.masks.data if detection_results.masks is not None else []
        boxes = detection_results.boxes.data if detection_results.boxes is not None else []
        
        print(f"🔍 YOLO detected {len(masks)} objects with {len(boxes)} boxes")
        
        # Process each detection
        processed_detections = []
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            try:
                # Extract box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                
                # Convert mask to proper format and resize to match image dimensions
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                
                # Resize mask to match original image dimensions
                if mask_np.shape != (img_height, img_width):
                    mask_np = cv2.resize(mask_np, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    print(f"   🔄 Resized mask from {mask.shape} to {mask_np.shape}")
                
                # Extract traits using the trait extractor
                traits = pipeline.trait_extractor.extract_advanced_traits(mask_np)
                
                detection_info = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_name': 'cucumber',
                    'traits': traits
                }
                processed_detections.append(detection_info)
                print(f"✅ Processed detection {i+1}: bbox={detection_info['bbox']}, conf={conf:.2f}")
                
            except Exception as e:
                print(f"⚠️ Failed to process detection {i+1}: {e}")
                continue
        
        # Create results structure (without numpy arrays)
        processed_results = {
            'detections': processed_detections,
            'cucumbers_found': len(processed_detections),
            'ruler_detected': False,  # We'll add ruler detection later if needed
            'ppcm': None,
            'traits': [
                {
                    'length_cm': float(d.get('traits', {}).get('length_px', 0)),
                    'width_cm': float(d.get('traits', {}).get('width_px', 0)),
                    'area_cm2': float(d.get('traits', {}).get('area_px', 0)),
                    'aspect_ratio': float(d.get('traits', {}).get('aspect_ratio', 0)),
                }
                for d in processed_detections
            ],
            'processing_time': None,
            'errors': []
        }
        
        print(f"🔍 Processed results: {len(processed_results['detections'])} detections")
        return True, processed_results
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error processing image: {str(e)}"

def create_visualization(image_array, results):
    """Create visualization of results with bounding boxes and trait information"""
    try:
        print(f"🔍 Creating visualization for {len(results.get('detections', []))} detections")
        
        # Debug: print first detection structure
        if results.get('detections'):
            first_detection = results['detections'][0]
            print(f"📋 First detection structure: {first_detection}")
            print(f"🔍 All detection keys: {list(first_detection.keys())}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Original image
        ax1.imshow(image_array)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Results visualization with bounding boxes and trait info
        ax2.imshow(image_array)
        ax2.set_title('Detection Results with Measurements')
        ax2.axis('off')
        
        # Draw bounding boxes and add trait information
        for i, detection in enumerate(results.get('detections', [])):
            try:
                bbox = detection.get('bbox')
                confidence = detection.get('confidence', 0)
                traits = detection.get('traits', {})
                
                if bbox and len(bbox) == 4:
                    # Draw bounding box
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         fill=False, edgecolor='red', linewidth=3)
                    ax2.add_patch(rect)
                    
                    # Add label with confidence
                    label = f"Cucumber {i+1}: {confidence:.2f}"
                    ax2.text(x1, max(0, y1 - 10), label,
                             color='red', fontsize=12, weight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    # Add trait measurements near the detection
                    if traits:
                        length = traits.get('length_px', 0)
                        width = traits.get('width_px', 0)
                        area = traits.get('area_px', 0)
                        
                        trait_text = f"L: {length:.0f}px\nW: {width:.0f}px\nA: {area:.0f}px²"
                        ax2.text(x2 + 5, y1, trait_text,
                                 color='blue', fontsize=10, weight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                    
                print(f"✅ Drew detection {i+1}: bbox={bbox}, conf={confidence:.2f}")
                
            except Exception as e:
                print(f"⚠️ Failed to draw detection {i+1}: {e}")
                continue
        
        # Add summary text
        summary_text = f"Detected: {len(results.get('detections', []))} cucumbers"
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
                fontsize=14, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        print("✅ Visualization created successfully with bounding boxes and measurements")
        return img_buffer
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple text-based visualization if image creation fails
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, f'Results: {len(results.get("detections", []))} detections\nVisualization failed: {str(e)}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            return img_buffer
        except:
            return None

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Decode image and also save to a temp file so pipeline can use path-based method
        file_bytes = file.read()
        image_array = np.frombuffer(file_bytes, np.uint8)
        image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_array is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_path = tmp.name
            # Save RGB back to BGR for cv2.imwrite
            cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        success, results = process_image_file(temp_path)

        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

        if not success:
            return jsonify({'error': results}), 500

        # Add visualization
        viz_buffer = create_visualization(image_rgb, results)
        if viz_buffer:
            results['visualization'] = base64.b64encode(viz_buffer.getvalue()).decode('utf-8')

        return jsonify(results)
    except Exception as e:
        print(f"❌ Upload handler error: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    success, message = load_model()
    return jsonify({'success': success, 'message': message})

@app.route('/model_status')
def model_status():
    return jsonify({
        'model_loaded': model_loaded,
        'model_path': "outputs/models/train/weights/best.pt" if model_loaded else None
    })

if __name__ == '__main__':
    print("🚀 Starting Cucumber HTP Web Interface...")
    print("📁 Looking for model at: outputs/models/train/weights/best.pt")

    success, message = load_model()
    if success:
        print("✅ Model loaded successfully!")
    else:
        print(f"⚠️  Model not loaded: {message}")
        print("💡 You can load the model through the web interface")

    print("🌐 Starting web server at http://localhost:5004")
    print("📱 Open your browser and upload images to test the model!")

    app.run(debug=True, host='0.0.0.0', port=5004)
