# YOLO12 Quick Start Guide for Cucumber Trait Extraction

## 🚀 Why YOLO12?

YOLO12 is the latest and most advanced YOLO model, offering significant improvements over YOLOv8:

- **Better Accuracy**: Higher mAP scores across all model scales
- **Attention-Centric Architecture**: Improved feature extraction with area attention
- **Enhanced Efficiency**: Fewer parameters while maintaining performance
- **Latest Innovations**: FlashAttention support, R-ELAN blocks, optimized MLP ratios

## 📋 Prerequisites

✅ **Python Environment**: Python 3.8+ with all dependencies installed  
✅ **Dataset**: Your Roboflow annotated cucumber dataset  
✅ **Hardware**: CPU or GPU (GPU recommended for faster training)  
✅ **Dependencies**: `pip install -r requirements.txt`  

## 🎯 Quick Start Commands

### 1. **Basic Training (Nano Model)**
```bash
python scripts/train_yolo12.py
```

### 2. **Custom Model Size**
```bash
# Small model (better accuracy, slower)
python scripts/train_yolo12.py --model-size s

# Medium model (balanced)
python scripts/train_yolo12.py --model-size m

# Large model (high accuracy)
python scripts/train_yolo12.py --model-size l

# Extra-large model (best accuracy, slowest)
python scripts/train_yolo12.py --model-size x
```

### 3. **Custom Training Parameters**
```bash
python scripts/train_yolo12.py \
    --model-size m \
    --epochs 150 \
    --batch-size 8 \
    --image-size 832 \
    --device 0
```

### 4. **Export After Training**
```bash
python scripts/train_yolo12.py --export
```

## 🔧 Configuration Options

### **Model Sizes**
| Size | Parameters | Speed | Accuracy | Use Case |
|------|------------|-------|----------|----------|
| **n** (nano) | ~3.2M | Fastest | Good | Quick prototyping |
| **s** (small) | ~11.4M | Fast | Better | Production ready |
| **m** (medium) | ~20.1M | Medium | High | Best balance |
| **l** (large) | ~26.4M | Slow | Higher | High accuracy needed |
| **x** (extra-large) | ~59.1M | Slowest | Highest | Research/benchmarks |

### **Key Parameters**
- `--epochs`: Training duration (default: 100)
- `--batch-size`: Batch size (default: 16, reduce if OOM)
- `--image-size`: Input resolution (default: 640, higher = better accuracy)
- `--device`: Training device (cpu, 0, 1, 2, 3, or auto)

## 📊 Expected Performance

Based on [YOLO12 benchmarks](https://docs.ultralytics.com/models/yolo12/#detection-performance-coco-val2017):

| Model | mAP50-95 | Speed (T4) | Parameters |
|-------|----------|------------|------------|
| YOLO12n | 40.6 | 2.87ms | 3.2M |
| YOLO12s | 47.2 | 4.86ms | 11.4M |
| YOLO12m | 51.3 | 6.77ms | 20.1M |
| YOLO12l | 53.7 | 8.89ms | 26.4M |
| YOLO12x | 55.2 | 11.79ms | 59.1M |

## 🎨 Dataset Structure

Your Roboflow dataset should be organized as:
```
data/
├── annotations/
│   ├── train/
│   │   ├── images/     # Training images
│   │   └── labels/     # Training annotations
│   ├── valid/
│   │   ├── images/     # Validation images
│   │   └── labels/     # Validation annotations
│   └── test/
│       ├── images/     # Test images
│       └── labels/     # Test annotations
└── data.yaml           # Dataset configuration
```

## 🚀 Training Process

### **Phase 1: Model Loading**
- YOLO12 model automatically downloads
- Pre-trained weights loaded for transfer learning

### **Phase 2: Training**
- Automatic dataset validation
- Progressive learning rate scheduling
- Real-time metrics monitoring

### **Phase 3: Validation**
- Automatic model evaluation
- Performance metrics calculation
- Best model selection

## 📈 Monitoring Training

Training progress is saved in:
```
models/yolo12/exp/
├── weights/
│   ├── best.pt         # Best model
│   └── last.pt         # Latest checkpoint
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
└── labels_correlogram.jpg # Label correlations
```

## 🔍 Troubleshooting

### **Out of Memory (OOM)**
```bash
# Reduce batch size
python scripts/train_yolo12.py --batch-size 8

# Use smaller model
python scripts/train_yolo12.py --model-size n

# Reduce image size
python scripts/train_yolo12.py --image-size 512
```

### **Slow Training**
```bash
# Use GPU if available
python scripts/train_yolo12.py --device 0

# Reduce image size
python scripts/train_yolo12.py --image-size 512

# Use smaller model
python scripts/train_yolo12.py --model-size n
```

### **Poor Accuracy**
```bash
# Increase epochs
python scripts/train_yolo12.py --epochs 200

# Use larger model
python scripts/train_yolo12.py --model-size l

# Increase image size
python scripts/train_yolo12.py --image-size 832
```

## 🎯 Next Steps After Training

### **1. Test Your Model**
```bash
python scripts/extract_traits.py \
    --model models/yolo12/exp/weights/best.pt \
    --image data/annotations/test/images/sample.jpg
```

### **2. Batch Inference**
```bash
python scripts/extract_traits.py \
    --model models/yolo12/exp/weights/best.pt \
    --image-dir data/annotations/test/images/
```

### **3. Analyze Results**
- Check training report: `models/yolo12/exp/yolo12_training_report.txt`
- Review validation metrics
- Test on new images

## 🌟 Advanced Features

### **FlashAttention (GPU Only)**
If you have compatible NVIDIA GPU:
```bash
# Enable in configs/yolo12_training_config.yaml
yolo12_optimizations:
  use_flash_attention: true
```

### **Custom Augmentation**
Modify `configs/yolo12_training_config.yaml`:
```yaml
augmentation:
  hsv_h: 0.02    # Increase hue variation
  hsv_s: 0.8     # Increase saturation variation
  degrees: 5.0   # Allow slight rotation
```

## 📚 Additional Resources

- [YOLO12 Official Documentation](https://docs.ultralytics.com/models/yolo12/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLO12 Paper](https://arxiv.org/abs/2502.12524)

## 🎉 Success Metrics

Your YOLO12 model is ready when:
- **mAP50 > 0.7** (70% accuracy)
- **Training loss stabilizes**
- **Validation metrics improve**
- **No overfitting detected**

---

**Ready to train?** Start with:
```bash
python scripts/train_yolo12.py --model-size m --epochs 100
```

This will give you a balanced model with good accuracy and reasonable training time!
