#!/usr/bin/env python3
"""
Production Training Script
Enhanced training configuration for production-ready cucumber detection model
"""

import os
import yaml
from pathlib import Path
import argparse
from ultralytics import YOLO

class ProductionTrainer:
    def __init__(self, data_yaml_path, model_size='yolo12s'):
        """Initialize production trainer."""
        self.data_yaml_path = Path(data_yaml_path)
        self.model_size = model_size
        
        # Production training parameters
        self.training_config = {
            'task': 'detect',
            'mode': 'train',
            'model': f'{model_size}.pt',
            'data': str(self.data_yaml_path.absolute()),
            'epochs': 500,  # Increased from 150
            'patience': 50,  # Increased patience
            'batch': 4,      # Reduced for CPU training
            'imgsz': 640,
            'save': True,
            'save_period': 25,  # Save every 25 epochs
            'cache': True,       # Enable caching for faster training
            'device': 'cpu',      # Use CPU for training
            'workers': 8,
            'project': 'models/production',
            'name': 'cucumber_traits_v2',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',  # Better optimizer
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,      # Cosine learning rate scheduling
            'close_mosaic': 10,
            'resume': False,
            'amp': True,         # Mixed precision training
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': True,  # Multi-scale training
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.1,      # Add dropout for regularization
            'val': True,
            'split': 'val',
            'save_json': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': True,        # Use FP16 for faster training
            'dnn': False,
            'plots': True,
            'source': None,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': True,     # Enable augmentation
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'embed': None,
            'show': False,
            'save_frames': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'show_boxes': True,
            'line_width': None,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': True,
            'opset': None,
            'workspace': None,
            'nms': False,
            'lr0': 0.001,        # Lower initial learning rate
            'lrf': 0.01,         # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5.0,  # Increased warmup
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'nbs': 64,
            'hsv_h': 0.015,      # Color augmentation
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,     # Rotation augmentation
            'translate': 0.2,     # Translation augmentation
            'scale': 0.9,        # Scale augmentation
            'shear': 2.0,        # Shear augmentation
            'perspective': 0.001, # Perspective augmentation
            'flipud': 0.1,       # Vertical flip
            'fliplr': 0.5,       # Horizontal flip
            'bgr': 0.0,
            'mosaic': 1.0,       # Mosaic augmentation
            'mixup': 0.3,        # Mixup augmentation
            'cutmix': 0.3,       # CutMix augmentation
            'copy_paste': 0.1,   # Copy-paste augmentation
            'copy_paste_mode': 'flip',
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'cfg': None,
            'tracker': 'botsort.yaml',
            'save_dir': 'models/production/cucumber_traits_v2'
        }
    
    def create_training_yaml(self, output_path):
        """Create training configuration YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.training_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Training configuration saved: {output_path}")
        return output_path
    
    def start_training(self, config_path=None):
        """Start the training process."""
        if config_path is None:
            config_path = self.create_training_yaml('configs/production_training.yaml')
        
        print("🚀 Starting production training...")
        print("=" * 60)
        print(f"📁 Data config: {self.data_yaml_path}")
        print(f"🏗️ Model: {self.model_size}")
        print(f"⏱️ Epochs: {self.training_config['epochs']}")
        print(f"📦 Batch size: {self.training_config['batch']}")
        print(f"🎯 Device: {self.training_config['device']}")
        print(f"📈 Learning rate: {self.training_config['lr0']}")
        print(f"🔄 Augmentation: Enabled")
        print("=" * 60)
        
        # Load model
        model = YOLO(self.training_config['model'])
        
        # Start training
        try:
            results = model.train(
                data=str(self.data_yaml_path),
                epochs=self.training_config['epochs'],
                patience=self.training_config['patience'],
                batch=self.training_config['batch'],
                imgsz=self.training_config['imgsz'],
                save_period=self.training_config['save_period'],
                cache=self.training_config['cache'],
                device=self.training_config['device'],
                workers=self.training_config['workers'],
                project=self.training_config['project'],
                name=self.training_config['name'],
                exist_ok=self.training_config['exist_ok'],
                pretrained=self.training_config['pretrained'],
                optimizer=self.training_config['optimizer'],
                verbose=self.training_config['verbose'],
                seed=self.training_config['seed'],
                deterministic=self.training_config['deterministic'],
                cos_lr=self.training_config['cos_lr'],
                close_mosaic=self.training_config['close_mosaic'],
                amp=self.training_config['amp'],
                multi_scale=self.training_config['multi_scale'],
                dropout=self.training_config['dropout'],
                half=self.training_config['half'],
                augment=self.training_config['augment'],
                degrees=self.training_config['degrees'],
                translate=self.training_config['translate'],
                scale=self.training_config['scale'],
                shear=self.training_config['shear'],
                perspective=self.training_config['perspective'],
                flipud=self.training_config['flipud'],
                fliplr=self.training_config['fliplr'],
                mosaic=self.training_config['mosaic'],
                mixup=self.training_config['mixup'],
                cutmix=self.training_config['cutmix'],
                copy_paste=self.training_config['copy_paste'],
                auto_augment=self.training_config['auto_augment'],
                erasing=self.training_config['erasing'],
                lr0=self.training_config['lr0'],
                lrf=self.training_config['lrf'],
                warmup_epochs=self.training_config['warmup_epochs'],
                weight_decay=self.training_config['weight_decay']
            )
            
            print("✅ Training completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def create_training_script(self, output_path):
        """Create a bash script for training."""
        script_content = f"""#!/bin/bash
# Production Training Script for Cucumber Detection
# Generated automatically

echo "🚀 Starting Production Training..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start training
python3 -m ultralytics train \\
    data={self.data_yaml_path.absolute()} \\
    model={self.model_size}.pt \\
    epochs={self.training_config['epochs']} \\
    patience={self.training_config['patience']} \\
    batch={self.training_config['batch']} \\
    imgsz={self.training_config['imgsz']} \\
    save_period={self.training_config['save_period']} \\
    cache=True \\
    device=0 \\
    workers={self.training_config['workers']} \\
    project={self.training_config['project']} \\
    name={self.training_config['name']} \\
    exist_ok=True \\
    pretrained=True \\
    optimizer={self.training_config['optimizer']} \\
    verbose=True \\
    seed={self.training_config['seed']} \\
    cos_lr=True \\
    amp=True \\
    multi_scale=True \\
    dropout={self.training_config['dropout']} \\
    half=True \\
    augment=True \\
    degrees={self.training_config['degrees']} \\
    translate={self.training_config['translate']} \\
    scale={self.training_config['scale']} \\
    shear={self.training_config['shear']} \\
    perspective={self.training_config['perspective']} \\
    flipud={self.training_config['flipud']} \\
    fliplr={self.training_config['fliplr']} \\
    mosaic={self.training_config['mosaic']} \\
    mixup={self.training_config['mixup']} \\
    cutmix={self.training_config['cutmix']} \\
    copy_paste={self.training_config['copy_paste']} \\
    auto_augment={self.training_config['auto_augment']} \\
    erasing={self.training_config['erasing']} \\
    lr0={self.training_config['lr0']} \\
    lrf={self.training_config['lrf']} \\
    warmup_epochs={self.training_config['warmup_epochs']} \\
    weight_decay={self.training_config['weight_decay']}

echo "✅ Training completed!"
"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        print(f"✅ Training script created: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Production Training Setup')
    parser.add_argument('--data-yaml', required=True, help='Path to data.yaml file')
    parser.add_argument('--model-size', default='yolo12s', choices=['yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'], help='YOLO model size')
    parser.add_argument('--create-config', action='store_true', help='Create training configuration file')
    parser.add_argument('--create-script', action='store_true', help='Create training bash script')
    parser.add_argument('--start-training', action='store_true', help='Start training immediately')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ProductionTrainer(args.data_yaml, args.model_size)
    
    if args.create_config:
        trainer.create_training_yaml('configs/production_training.yaml')
    
    if args.create_script:
        trainer.create_training_script('scripts/run_production_training.sh')
    
    if args.start_training:
        trainer.start_training()
    
    if not any([args.create_config, args.create_script, args.start_training]):
        print("ℹ️ Use --create-config, --create-script, or --start-training to proceed")
        print("📋 Example: python3 scripts/production_training.py --data-yaml data/clean_dataset/data.yaml --create-config --create-script")

if __name__ == "__main__":
    main()
