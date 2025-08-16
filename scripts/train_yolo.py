#!/usr/bin/env python3
"""
Training script for YOLO model on cucumber trait extraction dataset.
Handles dataset preparation and model training.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.yolo_trainer import YOLOTrainer


def main():
    """Main function for YOLO training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model for cucumber trait extraction"
    )
    
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to training configuration file (.yaml)"
    )
    
    parser.add_argument(
        "--raw-images", 
        help="Directory containing raw cucumber images"
    )
    
    parser.add_argument(
        "--annotations", 
        help="Directory containing YOLO format annotations"
    )
    
    parser.add_argument(
        "--prepare-only", 
        action="store_true",
        help="Only prepare dataset, don't train model"
    )
    
    parser.add_argument(
        "--train-only", 
        action="store_true",
        help="Only train model, don't prepare dataset"
    )
    
    parser.add_argument(
        "--export", 
        action="store_true",
        help="Export model to different formats after training"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="models",
        help="Output directory for trained models (default: models)"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Initialize trainer
    try:
        trainer = YOLOTrainer(args.config)
        print(f"YOLO trainer initialized with config: {args.config}")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        sys.exit(1)
    
    # Prepare dataset if requested
    if not args.train_only and args.raw_images and args.annotations:
        if not os.path.exists(args.raw_images):
            print(f"Error: Raw images directory not found: {args.raw_images}")
            sys.exit(1)
        
        if not os.path.exists(args.annotations):
            print(f"Error: Annotations directory not found: {args.annotations}")
            sys.exit(1)
        
        print("Preparing dataset for training...")
        try:
            trainer.prepare_dataset(args.raw_images, args.annotations)
            print("Dataset preparation completed successfully!")
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            sys.exit(1)
    
    # Train model if requested
    if not args.prepare_only:
        print("Starting model training...")
        try:
            # Train the model
            best_model = trainer.train_model(output_dir=args.output_dir)
            print(f"Training completed! Best model: {best_model}")
            
            # Validate the model
            print("Validating trained model...")
            validation_results = trainer.validate_model(best_model)
            
            # Export model if requested
            if args.export:
                print("Exporting model to different formats...")
                exported_paths = trainer.export_model(best_model)
                print(f"Model exported to: {exported_paths}")
            
            # Create training report
            print("Creating training report...")
            training_report = trainer.create_training_report(
                {}, validation_results, best_model
            )
            
            # Save training report
            report_path = os.path.join(args.output_dir, "training_report.txt")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(training_report)
            
            print(f"Training report saved to: {report_path}")
            
            # Print summary
            print("\n=== TRAINING SUMMARY ===")
            print(f"Best model: {best_model}")
            print(f"Validation mAP50: {validation_results.get('mAP50', 0):.4f}")
            print(f"Validation mAP50-95: {validation_results.get('mAP50-95', 0):.4f}")
            print(f"Precision: {validation_results.get('precision', 0):.4f}")
            print(f"Recall: {validation_results.get('recall', 0):.4f}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)
    
    print("All operations completed successfully!")


if __name__ == "__main__":
    main()
