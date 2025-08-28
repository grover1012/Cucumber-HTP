#!/usr/bin/env python3
"""
Cucumber High-Throughput Phenotyping (HTP) Pipeline Driver
Inspired by TomatoScanner's approach for robust phenotyping.

This script provides a single command interface for:
1. Training edge-aware YOLO segmentation models
2. Running inference with the complete phenotyping pipeline
3. Managing dataset preparation and validation

Usage:
    python src/run_pipeline.py --stage train --model yolov8m-seg.pt --epochs 150
    python src/run_pipeline.py --stage infer --img-dir data/test/images
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from train_seg import train, create_edge_training_config
from infer_seg import run_folder_pipeline

def validate_dataset(data_yaml: str) -> Dict[str, Any]:
    """
    Validate dataset configuration and structure.
    
    Args:
        data_yaml (str): Path to data.yaml file
    
    Returns:
        Dict[str, Any]: Dataset validation results
    """
    print("🔍 Validating dataset...")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")
    
    # Load data.yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    validation_results = {
        'data_yaml_path': data_yaml,
        'config_valid': True,
        'splits': {},
        'classes': data_config.get('names', []),
        'nc': data_config.get('nc', 0),
        'errors': []
    }
    
    # Validate required fields
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in data_config:
            validation_results['errors'].append(f"Missing required field: {field}")
            validation_results['config_valid'] = False
    
    # Validate splits
    for split in ['train', 'val', 'test']:
        if split in data_config:
            split_path = data_config[split]
            images_dir = os.path.join(os.path.dirname(data_yaml), split_path)
            labels_dir = os.path.join(os.path.dirname(data_yaml), split_path.replace('images', 'labels'))
            
            split_info = {
                'path': split_path,
                'images_dir': images_dir,
                'labels_dir': labels_dir,
                'images_exist': os.path.exists(images_dir),
                'labels_exist': os.path.exists(labels_dir)
            }
            
            if split_info['images_exist']:
                # Count images
                image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
                split_info['image_count'] = len(image_files)
            else:
                split_info['image_count'] = 0
                validation_results['errors'].append(f"Images directory not found: {images_dir}")
            
            if split_info['labels_exist']:
                # Count labels
                label_files = list(Path(labels_dir).glob("*.txt"))
                split_info['label_count'] = len(label_files)
            else:
                split_info['label_count'] = 0
                validation_results['errors'].append(f"Labels directory not found: {labels_dir}")
            
            validation_results['splits'][split] = split_info
    
    # Print validation summary
    print(f"📊 Dataset Validation Results:")
    print(f"  • Classes: {validation_results['nc']} ({', '.join(validation_results['classes'])})")
    
    for split_name, split_info in validation_results['splits'].items():
        print(f"  • {split_name.capitalize()}: {split_info['image_count']} images, {split_info['label_count']} labels")
    
    if validation_results['errors']:
        print(f"  ⚠️ Validation errors:")
        for error in validation_results['errors']:
            print(f"    - {error}")
        validation_results['config_valid'] = False
    else:
        print(f"  ✅ Dataset validation passed!")
    
    return validation_results

def prepare_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Prepare training configuration based on command line arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments
    
    Returns:
        Dict[str, Any]: Training configuration
    """
    print("⚙️ Preparing training configuration...")
    
    # Create base configuration
    config = create_edge_training_config(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        edge_lambda=args.edge_lambda,
        edgeboost_prob=args.edgeboost_prob,
        lambda_c=args.lambda_c,
        lambda_a=args.lambda_a
    )
    
    # Override with command line arguments
    if args.device != "auto":
        config['device'] = args.device
    
    if args.project:
        config['project'] = args.project
    
    if args.name:
        config['name'] = args.name
    
    # Save configuration
    config_path = os.path.join(config['project'], config['name'], 'training_config.yaml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📁 Training configuration saved to: {config_path}")
    
    return config

def run_training(args: argparse.Namespace) -> None:
    """
    Run the training stage.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    print("🚀 Starting Training Stage")
    print("=" * 50)
    
    # Validate dataset
    validation_results = validate_dataset(args.data)
    if not validation_results['config_valid']:
        print("❌ Dataset validation failed. Please fix the errors above.")
        return
    
    # Prepare training configuration
    config = prepare_training_config(args)
    
    # Start training
    try:
        results = train(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            edge_lambda=args.edge_lambda,
            edgeboost_prob=args.edgeboost_prob,
            lambda_c=args.lambda_c,
            lambda_a=args.lambda_a,
            device=args.device,
            project=args.project,
            name=args.name
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {os.path.join(args.project, args.name)}")
        
        # Save training summary
        summary_path = os.path.join(args.project, args.name, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Training summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

def run_inference(args: argparse.Namespace) -> None:
    """
    Run the inference stage.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    print("🔍 Starting Inference Stage")
    print("=" * 50)
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("💡 Please train a model first or specify the correct path.")
        return
    
    # Validate image directory
    if not os.path.exists(args.img_dir):
        print(f"❌ Image directory not found: {args.img_dir}")
        return
    
    # Run inference pipeline
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
        
        print("\n🎉 Inference completed successfully!")
        print(f"📊 Results saved to: {args.output_csv}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        sys.exit(1)

def create_manual_validation_template(output_path: str = "manual_validation_template.csv") -> None:
    """
    Create a template CSV file for manual validation.
    
    Args:
        output_path (str): Path to save the template
    """
    print("📝 Creating manual validation template...")
    
    template_data = {
        'image_name': ['IMG_001.jpg', 'IMG_002.jpg', 'IMG_003.jpg'],
        'cucumber_id': [0, 0, 1],
        'length_cm_gt': [14.2, 12.8, 16.5],
        'width_cm_gt': [3.2, 2.9, 3.8],
        'notes': ['Manual caliper measurement', 'Manual caliper measurement', 'Manual caliper measurement']
    }
    
    import pandas as pd
    df = pd.DataFrame(template_data)
    df.to_csv(output_path, index=False)
    
    print(f"📁 Manual validation template saved to: {output_path}")
    print("💡 Fill in your manual measurements and use this file for validation.")

def main():
    """Main pipeline driver function."""
    parser = argparse.ArgumentParser(
        description="Cucumber High-Throughput Phenotyping (HTP) Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train edge-aware segmentation model
  python src/run_pipeline.py --stage train --model yolov8m-seg.pt --epochs 150 --imgsz 1024 --batch 8
  
  # Run inference on test images
  python src/run_pipeline.py --stage infer --model outputs/models/yolov8m-seg-edge/weights/best.pt --img-dir data/test/images
  
  # Create manual validation template
  python src/run_pipeline.py --stage template
        """
    )
    
    # Main stage argument
    parser.add_argument("--stage", choices=["train", "infer", "template"], required=True,
                       help="Pipeline stage to run")
    
    # Training arguments
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument("--data", default="data/data.yaml",
                           help="Path to data.yaml file (default: data/data.yaml)")
    train_group.add_argument("--model", default="yolov8m-seg.pt",
                           help="Model to train (default: yolov8m-seg.pt)")
    train_group.add_argument("--epochs", type=int, default=150,
                           help="Number of training epochs (default: 150)")
    train_group.add_argument("--imgsz", type=int, default=1024,
                           help="Input image size (default: 1024)")
    train_group.add_argument("--batch", type=int, default=8,
                           help="Batch size (default: 8)")
    train_group.add_argument("--edge-lambda", type=float, default=0.3,
                           help="Weight for edge loss (default: 0.3)")
    train_group.add_argument("--edgeboost-prob", type=float, default=0.5,
                           help="Probability of applying EdgeBoost (default: 0.5)")
    train_group.add_argument("--lambda-c", type=float, default=1.5,
                           help="Contrast enhancement factor (default: 1.5)")
    train_group.add_argument("--lambda-a", type=float, default=1.6,
                           help="Acutance enhancement factor (default: 1.6)")
    train_group.add_argument("--device", default="auto",
                           help="Device to use for training (default: auto)")
    train_group.add_argument("--project", default="outputs/models",
                           help="Project directory (default: outputs/models)")
    train_group.add_argument("--name", default="",
                           help="Experiment name (default: auto-generated)")
    
    # Inference arguments
    infer_group = parser.add_argument_group("Inference Options")
    infer_group.add_argument("--img-dir", default="data/test/images",
                           help="Directory containing test images (default: data/test/images)")
    infer_group.add_argument("--output-csv", default="outputs/tables/phenotyping_results.csv",
                           help="Path to save CSV results (default: outputs/tables/phenotyping_results.csv)")
    infer_group.add_argument("--confidence", type=float, default=0.25,
                           help="Detection confidence threshold (default: 0.25)")
    infer_group.add_argument("--iou", type=float, default=0.5,
                           help="IoU threshold for NMS (default: 0.5)")
    infer_group.add_argument("--ruler-ticks-cm", type=float, default=1.0,
                           help="Expected distance between ruler ticks in cm (default: 1.0)")
    infer_group.add_argument("--no-viz", action="store_true",
                           help="Disable visualization saving")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print pipeline header
    print("🥒 Cucumber High-Throughput Phenotyping (HTP) Pipeline")
    print("🚀 Inspired by TomatoScanner's approach for robust phenotyping")
    print("=" * 70)
    
    # Run appropriate stage
    if args.stage == "train":
        run_training(args)
    elif args.stage == "infer":
        run_inference(args)
    elif args.stage == "template":
        create_manual_validation_template()
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
