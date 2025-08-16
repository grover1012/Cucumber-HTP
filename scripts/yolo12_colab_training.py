#!/usr/bin/env python3
"""
YOLO12 Training Script for Google Colab
Fast GPU training for cucumber trait extraction
"""

import os
import time
import zipfile
from pathlib import Path

def check_gpu():
    """Check GPU availability."""
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("❌ No GPU available. Please enable GPU in Colab!")
        print("Runtime → Change runtime type → Hardware accelerator → GPU")
        return False

def install_dependencies():
    """Install required packages."""
    print("📦 Installing dependencies...")
    os.system("pip install ultralytics -q")
    os.system("pip install roboflow -q")
    
    import ultralytics
    print(f"✅ Ultralytics version: {ultralytics.__version__}")

def create_dataset_yaml():
    """Create dataset.yaml for YOLO training."""
    import yaml
    
    dataset_config = {
        'train': 'data/annotations/train/images',
        'val': 'data/annotations/valid/images', 
        'test': 'data/annotations/test/images',
        'nc': 9,
        'names': [
            'big_ruler', 'blue_dot', 'color_chart', 'cucumber',
            'green_dot', 'label', 'objects', 'red_dot', 'ruler'
        ]
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("✅ Created dataset.yaml:")
    print(yaml.dump(dataset_config, default_flow_style=False))
    
    return dataset_config

def verify_dataset():
    """Verify dataset structure."""
    print("\n📊 Dataset verification:")
    
    for split in ['train', 'val', 'test']:
        img_path = f"data/annotations/{split}/images"
        if os.path.exists(img_path):
            num_images = len([f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split}: {num_images} images")
        else:
            print(f"  ❌ {split}: path not found")

def train_yolo12(model_size="s", epochs=150, batch_size=16, image_size=640):
    """Train YOLO12 model."""
    from ultralytics import YOLO
    
    print(f"\n🚀 Starting YOLO12{model_size} training...")
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Training configuration
    print(f"🎯 Training Configuration:")
    print(f"  Model: YOLO12{model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {image_size}")
    
    # Expected training time
    if model_size == "n":
        expected_time = "15-30 minutes"
    elif model_size == "s":
        expected_time = "30-60 minutes"
    elif model_size == "m":
        expected_time = "1-2 hours"
    else:
        expected_time = "2-4 hours"
    
    print(f"  Expected Time: {expected_time}")
    
    # Load YOLO12 model
    model = YOLO(f"yolo12{model_size}")
    print(f"✅ YOLO12{model_size} model loaded")
    
    # Start training
    start_time = time.time()
    
    results = model.train(
        data='dataset.yaml',
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=0,  # Use GPU
        project='cucumber_traits_yolo12',
        name=f'exp_{model_size}',
        save=True,
        plots=True,
        save_period=10,
        verbose=True
    )
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60  # minutes
    
    print(f"\n🎉 Training completed in {training_time:.1f} minutes!")
    print(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

def show_results(model_size="s"):
    """Show training results."""
    print("\n📊 Training Results:")
    print("=" * 50)
    
    best_model_path = f"cucumber_traits_yolo12/exp_{model_size}/weights/best.pt"
    last_model_path = f"cucumber_traits_yolo12/exp_{model_size}/weights/last.pt"
    
    if os.path.exists(best_model_path):
        print(f"🏆 Best Model: {best_model_path}")
        file_size = os.path.getsize(best_model_path) / (1024 * 1024)  # MB
        print(f"   Size: {file_size:.1f} MB")
    else:
        print(f"❌ Best model not found: {best_model_path}")
    
    if os.path.exists(last_model_path):
        print(f"📝 Last Model: {last_model_path}")
        file_size = os.path.getsize(last_model_path) / (1024 * 1024)  # MB
        print(f"   Size: {file_size:.1f} MB")
    else:
        print(f"❌ Last model not found: {last_model_path}")
    
    # List training outputs
    print(f"\n📁 Training outputs:")
    os.system(f"ls -la cucumber_traits_yolo12/exp_{model_size}/")
    
    # Show training plots
    print(f"\n📈 Training plots:")
    os.system(f"ls -la cucumber_traits_yolo12/exp_{model_size}/*.png")

def validate_model(model_size="s"):
    """Validate the trained model."""
    from ultralytics import YOLO
    
    print("\n🔍 Validating trained model...")
    
    best_model_path = f"cucumber_traits_yolo12/exp_{model_size}/weights/best.pt"
    
    if os.path.exists(best_model_path):
        # Load the trained model
        trained_model = YOLO(best_model_path)
        
        # Run validation
        val_results = trained_model.val(data='dataset.yaml')
        
        print("✅ Validation completed!")
        print(f"Results saved to: cucumber_traits_yolo12/exp_{model_size}/")
        return val_results
    else:
        print("❌ Best model not found. Cannot validate.")
        return None

def main():
    """Main training function."""
    print("🥒 YOLO12 Cucumber Trait Extraction Training")
    print("=" * 60)
    
    # Check GPU
    if not check_gpu():
        return
    
    # Install dependencies
    install_dependencies()
    
    # Create dataset.yaml
    dataset_config = create_dataset_yaml()
    
    # Verify dataset
    verify_dataset()
    
    # Training parameters
    MODEL_SIZE = "s"      # n, s, m, l, or x
    EPOCHS = 150
    BATCH_SIZE = 16       # Higher on GPU
    IMAGE_SIZE = 640
    
    # Start training
    results = train_yolo12(MODEL_SIZE, EPOCHS, BATCH_SIZE, IMAGE_SIZE)
    
    # Show results
    show_results(MODEL_SIZE)
    
    # Validate model
    validate_model(MODEL_SIZE)
    
    print("\n🎉 Training completed successfully!")
    print("\n📥 Next steps:")
    print("1. Download the trained model (.pt file)")
    print("2. Download the results ZIP file")
    print("3. Use the model for inference on new images")
    print("\n💡 To download files in Colab:")
    print("   from google.colab import files")
    print("   files.download('cucumber_traits_yolo12/exp_s/weights/best.pt')")

if __name__ == "__main__":
    main()
