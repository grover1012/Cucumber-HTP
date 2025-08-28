import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import our custom modules
from pose import rotate_to_major_axis, normalize_multiple_cucumbers
from scale_ruler import find_ppcm_simple, find_ppcm_robust, RulerDetector
from traits import CucumberTraitExtractor, extract_traits_advanced
from ocr_label import AccessionLabelReader, read_label_robust

class CucumberPhenotypingPipeline:
    """
    End-to-end cucumber phenotyping pipeline integrating all components.
    Inspired by TomatoScanner's approach for robust trait extraction.
    
    Args:
        model_path (str): Path to trained YOLO segmentation model
        confidence_threshold (float): Detection confidence threshold
        iou_threshold (float): IoU threshold for NMS
        ruler_tick_distance_cm (float): Expected distance between ruler ticks in cm
        save_visualizations (bool): Whether to save visualization images
        output_dir (str): Output directory for results
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25, 
                 iou_threshold: float = 0.5, ruler_tick_distance_cm: float = 1.0,
                 save_visualizations: bool = True, output_dir: str = "outputs"):
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.ruler_tick_distance_cm = ruler_tick_distance_cm
        self.save_visualizations = save_visualizations
        self.output_dir = output_dir
        
        # Initialize components
        self.model = None
        self.ruler_detector = None
        self.trait_extractor = None
        self.ocr_reader = None
        
        # Create output directories
        self._create_output_dirs()
        
        # Load model and initialize components
        self._initialize_components()
    
    def _create_output_dirs(self):
        """Create output directories for results."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "tables"), exist_ok=True)
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        print("🔧 Initializing pipeline components...")
        
        # Load YOLO model
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO model loaded: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
        
        # Initialize ruler detector
        self.ruler_detector = RulerDetector()
        print("✅ Ruler detector initialized")
        
        # Initialize trait extractor
        self.trait_extractor = CucumberTraitExtractor()
        print("✅ Trait extractor initialized")
        
        # Initialize OCR reader
        self.ocr_reader = AccessionLabelReader()
        print("✅ OCR reader initialized")
        
        print("🚀 Pipeline initialization complete!")
    
    def process_single_image(self, img_path: str) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            img_path (str): Path to input image
        
        Returns:
            Dict[str, Any]: Complete phenotyping results
        """
        print(f"📸 Processing image: {os.path.basename(img_path)}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        results = {
            'image_path': img_path,
            'image_name': os.path.basename(img_path),
            'cucumbers': [],
            'ruler_scale': None,
            'accession_label': None,
            'processing_metadata': {}
        }
        
        try:
            # Step 1: Detect ruler and compute scale
            print("  📏 Detecting ruler and computing scale...")
            try:
                ppcm = find_ppcm_simple(img, self.ruler_tick_distance_cm)
                if ppcm:
                    results['ruler_scale'] = ppcm
                    print(f"    ✅ Scale detected: {ppcm:.2f} pixels/cm")
                else:
                    print("    ⚠️ Scale detection failed, using pixel measurements only")
            except Exception as e:
                print(f"    ⚠️ Scale detection failed: {e}")
            
            # Step 2: Detect and segment cucumbers
            print("  🥒 Detecting and segmenting cucumbers...")
            detection_results = self.model.predict(
                source=img, 
                imgsz=1024, 
                conf=self.confidence_threshold, 
                iou=self.iou_threshold, 
                verbose=False
            )[0]
            
            if detection_results.masks is None:
                print("    ⚠️ No cucumbers detected")
                return results
            
            masks = detection_results.masks.data
            boxes = detection_results.boxes.data
            
            # Filter only cucumber class (ID: 4) and high confidence detections
            cucumber_masks = []
            cucumber_boxes = []
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                class_id = int(cls)
                
                # Based on investigation, the training data has WRONG labels:
                # - Class ID 4 (cucumber) contains rulers, charts, etc.
                # - Class IDs 1, 5, 9 (blue_dot, green_dot, red_dot) contain actual cucumbers
                # So we'll use those classes instead
                cucumber_class_ids = [1, 5, 9]  # blue_dot, green_dot, red_dot
                
                if class_id in cucumber_class_ids and conf > 0.3:
                    cucumber_masks.append(mask)
                    cucumber_boxes.append(box)
                    class_names = ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
                    print(f"    ✅ Processing cucumber {len(cucumber_masks)}: {class_name} (ID: {class_id}, conf: {conf:.3f})")
                else:
                    class_names = ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
                    print(f"    ⚠️  Filtered out: {class_name} (ID: {class_id}, conf: {conf:.3f}) - not a cucumber class")
            
            if not cucumber_masks:
                print("    ⚠️ No valid cucumbers detected after filtering")
                return results
            
            print(f"    ✅ Found {len(cucumber_masks)} valid cucumbers after filtering")
            
            # Step 3: Read accession label (once per image)
            print("  🏷️ Reading accession label...")
            try:
                accession = self.ocr_reader.read_accession_label(img)
                results['accession_label'] = accession
                if accession:
                    print(f"    ✅ Accession: {accession}")
                else:
                    print("    ⚠️ No accession label detected")
            except Exception as e:
                print(f"    ⚠️ OCR failed: {e}")
            
            # Step 4: Process each detected cucumber
            for i, mask in enumerate(cucumber_masks):
                print(f"  🔍 Processing cucumber {i+1}/{len(cucumber_masks)}...")
                
                cucumber_result = self._process_cucumber_mask(
                    img, mask, i, results['ruler_scale']
                )
                
                if cucumber_result:
                    results['cucumbers'].append(cucumber_result)
            
            # Step 5: Save visualizations if requested
            if self.save_visualizations:
                self._save_visualizations(img, results)
            
            print(f"  ✅ Processing complete: {len(results['cucumbers'])} cucumbers analyzed")
            
        except Exception as e:
            print(f"  ❌ Processing failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _process_cucumber_mask(self, img: np.ndarray, mask: np.ndarray, 
                              cucumber_id: int, ppcm: Optional[float]) -> Optional[Dict[str, Any]]:
        """
        Process a single cucumber mask to extract traits.
        
        Args:
            img (np.ndarray): Original image
            mask (np.ndarray): Binary mask for the cucumber
            cucumber_id (int): ID of the cucumber
            ppcm (Optional[float]): Pixels per cm conversion factor
        
        Returns:
            Optional[Dict[str, Any]]: Cucumber traits and measurements
        """
        try:
            # Convert mask to proper format
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            
            # Step 1: Pose normalization
            print(f"    🔄 Normalizing pose...")
            rotated_img, rotated_mask, rotation_angle = rotate_to_major_axis(img, mask_np)
            
            # Step 2: Extract traits from rotated mask
            print(f"    📊 Extracting traits...")
            traits = self.trait_extractor.extract_advanced_traits(rotated_mask)
            
            if not traits:
                print(f"    ⚠️ Failed to extract traits for cucumber {cucumber_id}")
                return None
            
            # Step 3: Convert to cm if scale is available
            if ppcm:
                cm_traits = self.trait_extractor.convert_to_cm(traits, ppcm)
                traits.update(cm_traits)
            
            # Step 4: Create result dictionary
            result = {
                'cucumber_id': cucumber_id,
                'traits_pixels': {k: v for k, v in traits.items() if k.endswith('_px')},
                'traits_cm': {k: v for k, v in traits.items() if k.endswith('_cm') or k.endswith('_cm2')},
                'traits_dimensionless': {k: v for k, v in traits.items() if not (k.endswith('_px') or k.endswith('_cm') or k.endswith('_cm2'))},
                'pose_metadata': {
                    'rotation_angle': float(rotation_angle),
                    'original_mask_area': int(np.sum(mask_np > 0)),
                    'rotated_mask_area': int(np.sum(rotated_mask > 0))
                }
            }
            
            # Add scale information
            if ppcm:
                result['scale_info'] = {
                    'pixels_per_cm': float(ppcm),
                    'scale_source': 'ruler_detection'
                }
            
            print(f"    ✅ Traits extracted: {len(traits)} measurements")
            return result
            
        except Exception as e:
            print(f"    ❌ Failed to process cucumber {cucumber_id}: {e}")
            return None
    
    def _save_visualizations(self, img: np.ndarray, results: Dict[str, Any]):
        """Save comprehensive visualization images with segmentation masks, bounding boxes, and measurements."""
        try:
            img_name = results['image_name']
            base_name = os.path.splitext(img_name)[0]
            
            # Create visualization with original image
            vis_img = img.copy()
            
            # Get detection results from the model
            detection_results = self.model.predict(
                source=img, 
                imgsz=1024, 
                conf=self.confidence_threshold, 
                iou=self.iou_threshold, 
                verbose=False
            )[0]
            
            # Extract masks and boxes
            masks = detection_results.masks.data if detection_results.masks is not None else []
            boxes = detection_results.boxes.data if detection_results.boxes is not None else []
            
            print(f"    🎨 Creating visualization with {len(masks)} masks and {len(boxes)} boxes")
            
            # Draw segmentation masks and bounding boxes
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                try:
                    # Extract box coordinates
                    x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                    
                    # Convert mask to proper format and resize to match image dimensions
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                    img_height, img_width = img.shape[:2]
                    
                    # Resize mask to match original image dimensions
                    if mask_np.shape != (img_height, img_width):
                        mask_np = cv2.resize(mask_np, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    
                    # Create colored overlay for segmentation mask
                    mask_overlay = np.zeros_like(vis_img)
                    mask_overlay[mask_np > 0] = [0, 255, 0]  # Green overlay
                    
                    # Apply mask overlay with transparency
                    alpha = 0.3
                    vis_img = cv2.addWeighted(vis_img, 1, mask_overlay, alpha, 0)
                    
                    # Draw bounding box
                    cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                    
                    # Add label with confidence
                    label = f"Cucumber {i+1}: {conf:.2f}"
                    cv2.putText(vis_img, label, (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    
                    # Add trait measurements if available
                    if i < len(results.get('cucumbers', [])):
                        cucumber = results['cucumbers'][i]
                        traits_cm = cucumber.get('traits_cm', {})
                        length_cm = traits_cm.get('length_cm', 'N/A')
                        width_cm = traits_cm.get('width_cm', 'N/A')
                        
                        if length_cm != 'N/A' and width_cm != 'N/A':
                            text = f"L: {length_cm:.1f}cm W: {width_cm:.1f}cm"
                            cv2.putText(vis_img, text, (int(x2) + 10, int(y1) + 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    print(f"      ✅ Drew detection {i+1}: bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                    
                except Exception as e:
                    print(f"      ⚠️ Failed to draw detection {i+1}: {e}")
                    continue
            
            # Add summary information
            summary_text = f"Detected: {len(masks)} cucumbers | Scale: {results.get('ruler_scale', 'N/A')} ppcm"
            cv2.putText(vis_img, summary_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Save visualization
            vis_path = os.path.join(self.output_dir, "visualizations", f"{base_name}_phenotyping.jpg")
            cv2.imwrite(vis_path, vis_img)
            
            print(f"    📸 Visualization saved: {vis_path}")
            
        except Exception as e:
            print(f"    ❌ Failed to save visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def process_folder(self, img_dir: str, output_csv: str = None) -> List[Dict[str, Any]]:
        """
        Process all images in a folder.
        
        Args:
            img_dir (str): Directory containing images
            output_csv (str): Path to save CSV results
        
        Returns:
            List[Dict[str, Any]]: List of results for all images
        """
        print(f"📁 Processing folder: {img_dir}")
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(img_dir).glob(f"*{ext}"))
            image_files.extend(Path(img_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {img_dir}")
        
        print(f"📸 Found {len(image_files)} images to process")
        
        # Process each image
        all_results = []
        for img_file in tqdm(image_files, desc="Processing images"):
            try:
                result = self.process_single_image(str(img_file))
                all_results.append(result)
            except Exception as e:
                print(f"❌ Failed to process {img_file}: {e}")
                all_results.append({
                    'image_path': str(img_file),
                    'image_name': img_file.name,
                    'error': str(e),
                    'cucumbers': []
                })
        
        # Save results to CSV
        if output_csv:
            self._save_results_csv(all_results, output_csv)
        
        # Save results to JSON
        json_path = os.path.join(self.output_dir, "tables", "phenotyping_results.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {json_path}")
        
        return all_results
    
    def _save_results_csv(self, results: List[Dict[str, Any]], output_csv: str):
        """Save results to CSV format for analysis."""
        rows = []
        
        for result in results:
            if 'error' in result:
                continue
            
            for cucumber in result['cucumbers']:
                row = {
                    'image_name': result['image_name'],
                    'accession_label': result['accession_label'],
                    'cucumber_id': cucumber['cucumber_id'],
                    'ruler_scale_ppcm': result.get('ruler_scale'),
                }
                
                # Add pixel traits
                for key, value in cucumber['traits_pixels'].items():
                    row[f"pixel_{key}"] = value
                
                # Add cm traits
                for key, value in cucumber['traits_cm'].items():
                    row[f"cm_{key}"] = value
                
                # Add dimensionless traits
                for key, value in cucumber['traits_dimensionless'].items():
                    row[key] = value
                
                # Add pose metadata
                for key, value in cucumber['pose_metadata'].items():
                    row[f"pose_{key}"] = value
                
                rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"📊 CSV results saved to: {output_csv}")
        
        # Print summary statistics
        if not df.empty:
            print("\n📈 Summary Statistics:")
            print(f"  • Total cucumbers analyzed: {len(df)}")
            print(f"  • Images processed: {df['image_name'].nunique()}")
            if 'cm_length_cm' in df.columns:
                print(f"  • Average length: {df['cm_length_cm'].mean():.2f} cm")
                print(f"  • Average width: {df['cm_width_cm'].mean():.2f} cm")
                print(f"  • Average aspect ratio: {df['aspect_ratio'].mean():.2f}")

def run_folder_pipeline(model_path: str = "outputs/models/yolov8m-seg-edge/weights/best.pt",
                       img_dir: str = "data/test/images",
                       output_csv: str = "outputs/tables/phenotyping_results.csv",
                       confidence: float = 0.25,
                       iou: float = 0.5,
                       ruler_tick_distance_cm: float = 1.0,
                       save_viz: bool = True):
    """
    Run the complete phenotyping pipeline on a folder of images.
    
    Args:
        model_path (str): Path to trained YOLO segmentation model
        img_dir (str): Directory containing test images
        output_csv (str): Path to save CSV results
        confidence (float): Detection confidence threshold
        iou (float): IoU threshold for NMS
        ruler_tick_distance_cm (float): Expected distance between ruler ticks in cm
        save_viz (bool): Whether to save visualization images
    """
    
    print("🚀 Starting Cucumber Phenotyping Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = CucumberPhenotypingPipeline(
        model_path=model_path,
        confidence_threshold=confidence,
        iou_threshold=iou,
        ruler_tick_distance_cm=ruler_tick_distance_cm,
        save_visualizations=save_viz,
        output_dir="outputs"
    )
    
    # Process folder
    results = pipeline.process_folder(img_dir, output_csv)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📊 Processed {len(results)} images")
    
    # Count total cucumbers
    total_cucumbers = sum(len(r.get('cucumbers', [])) for r in results)
    print(f"🥒 Analyzed {total_cucumbers} cucumbers")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cucumber phenotyping pipeline")
    parser.add_argument("--model", default="outputs/models/yolov8m-seg-edge/weights/best.pt", 
                       help="Path to trained YOLO segmentation model")
    parser.add_argument("--img-dir", default="data/test/images", 
                       help="Directory containing test images")
    parser.add_argument("--output-csv", default="outputs/tables/phenotyping_results.csv", 
                       help="Path to save CSV results")
    parser.add_argument("--confidence", type=float, default=0.25, 
                       help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, 
                       help="IoU threshold for NMS")
    parser.add_argument("--ruler-ticks-cm", type=float, default=1.0, 
                       help="Expected distance between ruler ticks in cm")
    parser.add_argument("--no-viz", action="store_true", 
                       help="Disable visualization saving")
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        results = run_folder_pipeline(
            model_path=args.model,
            img_dir=args.img_dir,
            output_csv=args.output_csv,
            confidence=args.confidence,
            iou=args.iou,
            ruler_tick_distance_cm=args.ruler_ticks_cm,
            save_viz=not args.no_viz
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {args.output_csv}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        exit(1)
