#!/usr/bin/env python3
"""
YOLO12 Training Script for Cucumber Trait Extraction

This script uses the latest YOLO12 model for optimal performance.
YOLO12 offers better accuracy and efficiency compared to YOLOv8.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.yolo_trainer import YOLOTrainer


def main():
    """Main function for YOLO12 training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO12 model for cucumber trait extraction"
    )
    
    parser.add_argument(
        "--config", 
        default="configs/yolo12_training_config.yaml",
        help="Path to YOLO12 training configuration file (default: configs/yolo12_training_config.yaml)"
    )
    
    parser.add_argument(
        "--model-size",
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help="YOLO12 model size: n(nano), s(small), m(medium), l(large), x(extra-large) (default: n)"
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        help="Path to pretrained YOLO12 weights to load before training",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="models/yolo12",
        help="Output directory for trained models (default: models/yolo12)"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use for training: cpu, 0, 1, 2, 3, or auto (default: auto)"
    )
    
    parser.add_argument(
        "--export", 
        action="store_true",
        help="Export model to different formats after training"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Using default YOLO12 configuration...")
        args.config = "configs/yolo12_training_config.yaml"
    
    # Initialize trainer
    try:
        trainer = YOLOTrainer(args.config)
        print(f"YOLO12 trainer initialized with config: {args.config}")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        sys.exit(1)
    
    # Update configuration with command line arguments
    trainer.config['training']['epochs'] = args.epochs
    trainer.config['training']['batch_size'] = args.batch_size
    trainer.config['training']['imgsz'] = args.image_size
    trainer.config['training']['device'] = args.device
    
    # Determine model architecture or pretrained weights
    if args.pretrained:
        model_architecture = args.pretrained
    else:
        model_architecture = f"yolo12{args.model_size}"

    # Store selected model path in configuration
    trainer.config['model'] = model_architecture
    
    print(f"\n=== YOLO12 Training Configuration ===")
    print(f"Model: {model_architecture}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.image_size}")
    print(f"Device: {args.device}")
    print(f"Output Directory: {args.output_dir}")
    
    # Check dataset structure
    print(f"\n=== Dataset Information ===")
    print(f"Classes: {trainer.class_names}")
    print(f"Number of classes: {trainer.num_classes}")
    
    # Verify dataset paths
    dataset_paths = [
        "data/annotations/train/images",
        "data/annotations/valid/images", 
        "data/annotations/test/images"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            num_files = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{path}: {num_files} images")
        else:
            print(f"Warning: {path} not found")
    
    # Start training
    print(f"\n=== Starting YOLO12 Training ===")
    try:
        # Train the model
        best_model = trainer.train_model(
            model_architecture=model_architecture,
            output_dir=args.output_dir,
            use_yolo12=True
        )
        print(f"Training completed! Best model: {best_model}")
        
        # Validate the model
        print("Validating trained YOLO12 model...")
        validation_results = trainer.validate_model(best_model)
        
        # Export model if requested
        if args.export:
            print("Exporting YOLO12 model to different formats...")
            exported_paths = trainer.export_model(best_model)
            print(f"Model exported to: {exported_paths}")
        
        # Create training report
        print("Creating YOLO12 training report...")
        training_report = trainer.create_training_report(
            {}, validation_results, best_model
        )
        
        # Save training report
        report_path = os.path.join(args.output_dir, "yolo12_training_report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(training_report)
        
        print(f"YOLO12 training report saved to: {report_path}")
        
        # Print summary
        print("\n=== YOLO12 TRAINING SUMMARY ===")
        print(f"Best model: {best_model}")
        print(f"Validation mAP50: {validation_results.get('mAP50', 0):.4f}")
        print(f"Validation mAP50-95: {validation_results.get('mAP50-95', 0):.4f}")
        print(f"Precision: {validation_results.get('precision', 0):.4f}")
        print(f"Recall: {validation_results.get('recall', 0):.4f}")
        
        print(f"\n🎉 YOLO12 training completed successfully!")
        print(f"Your model is ready for cucumber trait extraction!")
        
    except Exception as e:
        print(f"Error during YOLO12 training: {e}")
        sys.exit(1)
    
    print("\n=== Next Steps ===")
    print("1. Test your trained model:")
    print(f"   python scripts/extract_traits.py --model {best_model} --image data/annotations/test/images/")
    print("2. Run batch inference:")
    print(f"   python scripts/extract_traits.py --model {best_model} --image-dir data/annotations/test/images/")
    print("3. Check the training report for detailed metrics")


if __name__ == "__main__":
    main()
