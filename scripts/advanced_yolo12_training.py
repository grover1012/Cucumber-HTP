#!/usr/bin/env python3
"""
Advanced YOLO12 Training with Multi-Stage Fine-tuning
COCO Pre-trained → Agricultural Fine-tune → Cucumber Fine-tune
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

class AdvancedYOLO12Trainer:
    def __init__(self, config_path):
        """Initialize advanced trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        
    def _load_config(self, config_path):
        """Load training configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_coco_pretrained_model(self):
        """Load COCO pre-trained YOLO12 model."""
        print("🚀 Loading COCO pre-trained YOLO12 model...")
        
        model_size = self.config['model_size']
        model_path = f"yolo12{model_size}.pt"
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ Successfully loaded {model_path}")
            print(f"📊 Model info: {self.model.info()}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def stage1_coco_validation(self):
        """Stage 1: Validate COCO pre-trained model performance."""
        print("\n🎯 Stage 1: COCO Pre-trained Model Validation")
        print("=" * 50)
        
        if not self.model:
            print("❌ No model loaded. Please load COCO pre-trained model first.")
            return False
        
        # Test on a sample image to verify model is working
        test_image = "data/annotations/test/images/AM030_YF_2021_jpg.rf.541ac98d17535ed79e21ac842779f108.jpg"
        
        if os.path.exists(test_image):
            print(f"🧪 Testing COCO pre-trained model on: {test_image}")
            
            try:
                results = self.model(test_image, conf=0.25)
                print(f"✅ COCO model inference successful")
                print(f"📊 Detected objects: {len(results[0].boxes) if results[0].boxes else 0}")
                return True
            except Exception as e:
                print(f"❌ COCO model inference failed: {e}")
                return False
        else:
            print(f"⚠️ Test image not found: {test_image}")
            return True  # Continue anyway
    
    def stage2_agricultural_finetuning(self):
        """Stage 2: Fine-tune on agricultural datasets (optional)."""
        print("\n🌾 Stage 2: Agricultural Dataset Fine-tuning")
        print("=" * 50)
        
        if not self.config.get('enable_agricultural_finetuning', False):
            print("⏭️ Agricultural fine-tuning disabled. Skipping to Stage 3.")
            return True
        
        agricultural_dataset = self.config.get('agricultural_dataset_path')
        if not agricultural_dataset or not os.path.exists(agricultural_dataset):
            print("⚠️ Agricultural dataset not found. Skipping Stage 2.")
            return True
        
        print(f"🌾 Fine-tuning on agricultural dataset: {agricultural_dataset}")
        
        try:
            # Fine-tune with conservative learning rate
            results = self.model.train(
                data=agricultural_dataset,
                epochs=self.config['agricultural_epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch_size'],
                device=self.config['device'],
                lr0=self.config['learning_rate'] * 0.1,  # Lower learning rate
                patience=self.config['patience'],
                save_period=self.config['save_period'],
                project=self.config['project_name'],
                name="stage2_agricultural",
                exist_ok=True
            )
            
            print("✅ Agricultural fine-tuning completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Agricultural fine-tuning failed: {e}")
            return False
    
    def stage3_cucumber_finetuning(self):
        """Stage 3: Fine-tune on cucumber dataset."""
        print("\n🥒 Stage 3: Cucumber Dataset Fine-tuning")
        print("=" * 50)
        
        cucumber_dataset = self.config['cucumber_dataset_path']
        if not os.path.exists(cucumber_dataset):
            print(f"❌ Cucumber dataset not found: {cucumber_dataset}")
            return False
        
        print(f"🥒 Fine-tuning on cucumber dataset: {cucumber_dataset}")
        
        try:
            # Fine-tune with full learning rate
            results = self.model.train(
                data=cucumber_dataset,
                epochs=self.config['cucumber_epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch_size'],
                device=self.config['device'],
                lr0=self.config['learning_rate'],
                patience=self.config['patience'],
                save_period=self.config['save_period'],
                project=self.config['project_name'],
                name="stage3_cucumber",
                exist_ok=True,
                # Advanced training options
                cos_lr=True,  # Cosine learning rate scheduling
                warmup_epochs=3,  # Warmup epochs
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                # Data augmentation
                mosaic=1.0,
                mixup=0.1,
                copy_paste=0.1,
                # Regularization
                weight_decay=0.0005,
                dropout=0.1
            )
            
            print("✅ Cucumber fine-tuning completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Cucumber fine-tuning failed: {e}")
            return False
    
    def validate_model(self):
        """Validate the final fine-tuned model."""
        print("\n🔍 Model Validation")
        print("=" * 50)
        
        if not self.model:
            print("❌ No model loaded for validation.")
            return False
        
        cucumber_dataset = self.config['cucumber_dataset_path']
        
        try:
            # Run validation
            results = self.model.val(data=cucumber_dataset)
            
            print("✅ Model validation completed!")
            print(f"📊 Validation results: {results}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False
    
    def export_model(self):
        """Export the trained model in various formats."""
        print("\n📦 Model Export")
        print("=" * 50)
        
        if not self.model:
            print("❌ No model loaded for export.")
            return False
        
        try:
            # Export to different formats
            export_paths = self.model.export(format=['onnx', 'torchscript', 'tflite'])
            
            print("✅ Model exported successfully!")
            for path in export_paths:
                print(f"📁 Exported to: {path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model export failed: {e}")
            return False
    
    def run_complete_training(self):
        """Run the complete multi-stage training pipeline."""
        print("🚀 Advanced YOLO12 Training Pipeline")
        print("=" * 60)
        print("Strategy: COCO Pre-trained → Agricultural Fine-tune → Cucumber Fine-tune")
        print("=" * 60)
        
        # Stage 1: Load COCO pre-trained model
        if not self.load_coco_pretrained_model():
            return False
        
        # Stage 1: Validate COCO model
        if not self.stage1_coco_validation():
            print("⚠️ COCO validation failed, but continuing...")
        
        # Stage 2: Agricultural fine-tuning (optional)
        if not self.stage2_agricultural_finetuning():
            print("⚠️ Agricultural fine-tuning failed, but continuing...")
        
        # Stage 3: Cucumber fine-tuning
        if not self.stage3_cucumber_finetuning():
            return False
        
        # Validation
        if not self.validate_model():
            print("⚠️ Model validation failed, but continuing...")
        
        # Export
        if not self.export_model():
            print("⚠️ Model export failed, but continuing...")
        
        print("\n🎉 Advanced training pipeline completed!")
        print("🥒 Your robust cucumber detection model is ready!")
        
        return True

def main():
    """Main function for advanced YOLO12 training."""
    parser = argparse.ArgumentParser(description="Advanced YOLO12 Training with Multi-Stage Fine-tuning")
    parser.add_argument("--config", required=True, help="Path to training configuration file")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AdvancedYOLO12Trainer(args.config)
    
    # Run complete training pipeline
    success = trainer.run_complete_training()
    
    if success:
        print("\n🏆 Training completed successfully!")
        print("🎯 Your model is ready for laboratory deployment!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
