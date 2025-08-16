"""
YOLO training script for cucumber trait extraction.
Handles dataset preparation, training configuration, and model training.
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import random
import json
from ultralytics import YOLO


class YOLOTrainer:
    """
    YOLO trainer for cucumber trait extraction models.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.dataset_path = self.config['dataset']['path']
        self.class_names = self.config['dataset']['names']
        self.num_classes = self.config['dataset']['nc']
        
    def _load_config(self) -> Dict:
        """Load training configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def prepare_dataset(self, raw_images_dir: str, 
                       annotations_dir: str,
                       output_dir: str = "data") -> None:
        """
        Prepare dataset for YOLO training.
        
        Args:
            raw_images_dir: Directory containing raw images
            annotations_dir: Directory containing YOLO format annotations
            output_dir: Output directory for prepared dataset
        """
        print("Preparing dataset for YOLO training...")
        
        # Create output directories
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        test_dir = os.path.join(output_dir, "test")
        
        for dir_path in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(dir_path, "labels"), exist_ok=True)
        
        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(raw_images_dir).glob(f"*{ext}"))
            image_files.extend(Path(raw_images_dir).glob(f"*{ext.upper()}"))
        
        # Shuffle files for random split
        random.shuffle(image_files)
        
        # Calculate split indices
        total_files = len(image_files)
        train_split = int(total_files * self.config['dataset_split']['train'])
        val_split = int(total_files * self.config['dataset_split']['validation'])
        
        # Split files
        train_files = image_files[:train_split]
        val_files = image_files[train_split:train_split + val_split]
        test_files = image_files[train_split + val_split:]
        
        print(f"Total images: {total_files}")
        print(f"Train: {len(train_files)}")
        print(f"Validation: {len(val_files)}")
        print(f"Test: {len(test_files)}")
        
        # Copy files to respective directories
        self._copy_dataset_split(train_files, train_dir, annotations_dir)
        self._copy_dataset_split(val_files, val_dir, annotations_dir)
        self._copy_dataset_split(test_files, test_dir, annotations_dir)
        
        # Create dataset.yaml file
        self._create_dataset_yaml(output_dir)
        
        print("Dataset preparation completed!")
    
    def _copy_dataset_split(self, image_files: List[Path], 
                           output_dir: str, 
                           annotations_dir: str) -> None:
        """
        Copy images and annotations for a dataset split.
        
        Args:
            image_files: List of image files for this split
            output_dir: Output directory for this split
            annotations_dir: Directory containing annotations
        """
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        
        for image_file in image_files:
            # Copy image
            image_name = image_file.name
            image_dest = os.path.join(images_dir, image_name)
            shutil.copy2(image_file, image_dest)
            
            # Copy corresponding annotation
            annotation_name = image_file.stem + ".txt"
            annotation_src = os.path.join(annotations_dir, annotation_name)
            annotation_dest = os.path.join(labels_dir, annotation_name)
            
            if os.path.exists(annotation_src):
                shutil.copy2(annotation_src, annotation_dest)
            else:
                # Create empty annotation file if none exists
                with open(annotation_dest, 'w') as f:
                    pass
    
    def _create_dataset_yaml(self, output_dir: str) -> None:
        """
        Create dataset.yaml file for YOLO training.
        
        Args:
            output_dir: Dataset output directory
        """
        dataset_yaml = {
            'path': os.path.abspath(output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.num_classes,
            'names': self.class_names
        }
        
        yaml_path = os.path.join(output_dir, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to {yaml_path}")
    
    def train_model(self, model_architecture: str = None,
                   output_dir: str = "models",
                   use_yolo12: bool = True) -> str:
        """
        Train YOLO model.
        
        Args:
            model_architecture: YOLO model architecture to use
            output_dir: Output directory for trained models
            
        Returns:
            Path to best trained model
        """
        if model_architecture is None:
            model_architecture = self.config['model']['architecture']
        
        print(f"Starting training with {model_architecture}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model - YOLO12 models
        if use_yolo12 and model_architecture.startswith("yolo12"):
            print(f"Loading YOLO12 model: {model_architecture}")
            try:
                # YOLO12 models are automatically downloaded by Ultralytics
                # Use the full model name that Ultralytics recognizes
                model = YOLO(model_architecture)
                print(f"YOLO12 model loaded successfully: {model_architecture}")
            except Exception as e:
                print(f"Error loading YOLO12 model: {e}")
                print("Trying alternative approach...")
                # Try with just the base name
                base_name = model_architecture.replace("-seg.pt", "")
                model = YOLO(base_name)
                print(f"YOLO12 model loaded with base name: {base_name}")
        else:
            model = YOLO(model_architecture)
        
        # Training parameters
        train_params = {
            'data': os.path.join(self.dataset_path, 'dataset.yaml'),
            'epochs': self.config['training']['epochs'],
            'batch': self.config['training']['batch_size'],
            'imgsz': self.config['training']['imgsz'],
            'device': self.config['training']['device'],
            'project': output_dir,
            'name': 'cucumber_traits',
            'exist_ok': True,
            'patience': 20,  # Early stopping patience
            'save_period': self.config['validation']['save_period'],
            'plots': self.config['validation']['plots'],
            'save_json': self.config['validation']['save_json']
        }
        
        # Start training
        results = model.train(**train_params)
        
        # Get best model path
        best_model_path = results.best
        print(f"Training completed! Best model saved to: {best_model_path}")
        
        return best_model_path
    
    def validate_model(self, model_path: str, 
                      dataset_path: str = None) -> Dict:
        """
        Validate trained model.
        
        Args:
            model_path: Path to trained model
            dataset_path: Path to validation dataset
            
        Returns:
            Validation results
        """
        if dataset_path is None:
            dataset_path = os.path.join(self.dataset_path, 'dataset.yaml')
        
        print(f"Validating model: {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(data=dataset_path)
        
        # Extract metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1_score': results.box.map50 * 2 / (results.box.map50 + 1)
        }
        
        print("Validation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def export_model(self, model_path: str, 
                    export_formats: List[str] = None) -> List[str]:
        """
        Export model to different formats.
        
        Args:
            model_path: Path to trained model
            export_formats: List of export formats
            
        Returns:
            List of exported model paths
        """
        if export_formats is None:
            export_formats = ['onnx', 'torchscript', 'tflite']
        
        print(f"Exporting model to formats: {export_formats}")
        
        # Load model
        model = YOLO(model_path)
        
        exported_paths = []
        
        for format_name in export_formats:
            try:
                exported_path = model.export(format=format_name)
                exported_paths.append(exported_path)
                print(f"Exported to {format_name}: {exported_path}")
            except Exception as e:
                print(f"Failed to export to {format_name}: {e}")
        
        return exported_paths
    
    def create_training_report(self, training_results: Dict,
                              validation_results: Dict,
                              model_path: str) -> str:
        """
        Create a comprehensive training report.
        
        Args:
            training_results: Training results
            validation_results: Validation results
            model_path: Path to trained model
            
        Returns:
            Formatted training report
        """
        report = "=== YOLO TRAINING REPORT ===\n\n"
        
        # Model information
        report += "MODEL INFORMATION:\n"
        report += f"  Architecture: {self.config['model']['architecture']}\n"
        report += f"  Classes: {', '.join(self.class_names)}\n"
        report += f"  Number of classes: {self.num_classes}\n"
        report += f"  Model path: {model_path}\n\n"
        
        # Training configuration
        report += "TRAINING CONFIGURATION:\n"
        report += f"  Epochs: {self.config['training']['epochs']}\n"
        report += f"  Batch size: {self.config['training']['batch_size']}\n"
        report += f"  Image size: {self.config['training']['imgsz']}\n"
        report += f"  Device: {self.config['training']['device']}\n\n"
        
        # Training results
        if training_results:
            report += "TRAINING RESULTS:\n"
            for key, value in training_results.items():
                report += f"  {key}: {value}\n"
            report += "\n"
        
        # Validation results
        if validation_results:
            report += "VALIDATION RESULTS:\n"
            for metric, value in validation_results.items():
                report += f"  {metric}: {value:.4f}\n"
            report += "\n"
        
        # Dataset information
        report += "DATASET INFORMATION:\n"
        report += f"  Dataset path: {self.dataset_path}\n"
        report += f"  Train split: {self.config['dataset_split']['train']*100:.1f}%\n"
        report += f"  Validation split: {self.config['dataset_split']['validation']*100:.1f}%\n"
        report += f"  Test split: {self.config['dataset_split']['test']*100:.1f}%\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        if validation_results.get('mAP50', 0) > 0.8:
            report += "  ✓ Excellent model performance\n"
        elif validation_results.get('mAP50', 0) > 0.6:
            report += "  ✓ Good model performance\n"
        else:
            report += "  ⚠ Consider additional training or data augmentation\n"
        
        return report


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO model for cucumber trait extraction")
    parser.add_argument("--config", required=True, help="Path to training configuration file")
    parser.add_argument("--raw-images", help="Directory containing raw images")
    parser.add_argument("--annotations", help="Directory containing YOLO annotations")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset, don't train")
    parser.add_argument("--train-only", action="store_true", help="Only train model, don't prepare dataset")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOTrainer(args.config)
    
    # Prepare dataset if requested
    if not args.train_only and args.raw_images and args.annotations:
        trainer.prepare_dataset(args.raw_images, args.annotations)
    
    # Train model if requested
    if not args.prepare_only:
        best_model = trainer.train_model()
        
        # Validate model
        validation_results = trainer.validate_model(best_model)
        
        # Export model if requested
        if args.export:
            exported_paths = trainer.export_model(best_model)
        
        # Create training report
        training_report = trainer.create_training_report(
            {}, validation_results, best_model
        )
        
        # Save report
        report_path = os.path.join("models", "training_report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(training_report)
        
        print(f"Training report saved to {report_path}")


if __name__ == "__main__":
    main()
