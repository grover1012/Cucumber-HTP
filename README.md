# 🥒 Cucumber High-Throughput Phenotyping (HTP) Pipeline

A state-of-the-art computer vision pipeline for automated cucumber trait extraction using YOLO12 object detection and SAM2 segmentation.

## 🚀 Features

- **🎯 YOLO12 Object Detection**: Latest YOLO model for accurate cucumber detection
- **🔍 SAM2 Segmentation**: Meta's latest segmentation model for precise trait extraction
- **📊 Comprehensive Trait Extraction**: Length, width, curvature, color metrics, and more
- **🖼️ High-Quality Visualization**: Annotated images with detections and segmentation
- **⚡ Fast Processing**: Optimized for both CPU and GPU environments
- **🔧 Modular Design**: Easy to extend and customize for different use cases

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/grover1012/Cucumber-HTP.git
cd cucumber-htp
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download SAM2 model**
```bash
mkdir -p models
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -o models/sam2.1_hiera_tiny.pt
```

4. **Clone SAM2 repository**
```bash
git clone https://github.com/facebookresearch/sam2.git sam2
cd sam2
pip install -e .
cd ..
```

## 🚀 Quick Start

### 1. Test the Pipeline

```bash
# Test on a single image
python3 scripts/test_enhanced_extractor.py

# Test SAM2 directly
python3 scripts/test_sam2_direct.py
```

### 2. Process Your Own Images

```bash
# Single image processing
python3 src/inference/enhanced_trait_extractor.py \
    --model models/yolo12/cucumber_traits/weights/best.pt \
    --image path/to/your/image.jpg \
    --output-dir results/

# Batch processing
python3 src/inference/enhanced_trait_extractor.py \
    --model models/yolo12/cucumber_traits/weights/best.pt \
    --image-dir path/to/your/images/ \
    --output-dir results/batch/
```

## 📁 Project Structure

```
cucumber_HTP/
├── 📁 data/                          # Data directory
│   ├── 📁 raw_images/               # Original cucumber images
│   ├── 📁 annotations/              # Annotated dataset
│   │   ├── 📁 train/               # Training data
│   │   ├── 📁 valid/               # Validation data
│   │   └── 📁 test/                # Test data
│   └── 📁 processed/                # Processed images
├── 📁 models/                        # Model files
│   ├── 📁 yolo12/                  # YOLO12 trained models
│   └── 📁 sam2/                    # SAM2 models
├── 📁 src/                          # Source code
│   ├── 📁 preprocessing/            # Image preprocessing
│   ├── 📁 annotation/              # Annotation tools
│   ├── 📁 training/                # Training scripts
│   ├── 📁 inference/               # Inference pipeline
│   └── 📁 utils/                   # Utility functions
├── 📁 scripts/                      # Executable scripts
├── 📁 configs/                      # Configuration files
├── 📁 notebooks/                    # Jupyter notebooks
├── 📁 docs/                         # Documentation
└── 📁 results/                      # Output results
```

## 🎯 Usage

### Object Detection Classes

The pipeline detects 12 different object classes:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | big_ruler | Large ruler for calibration |
| 1 | blue_dot | Blue color reference |
| 2 | cavity | Cucumber cavity/indentation |
| 3 | color_chart | Color calibration chart |
| 4 | cucumber | Whole cucumber fruit |
| 5 | green_dot | Green color reference |
| 6 | hollow | Hollow/cavity in cucumber |
| 7 | label | Accession ID label |
| 8 | objects | Other objects |
| 9 | red_dot | Red color reference |
| 10 | ruler | Standard ruler |
| 11 | slice | Cucumber slice |

### Trait Extraction

The pipeline extracts comprehensive phenotypic traits:

- **📏 Size Metrics**: Length, width, aspect ratio, area, volume
- **🎨 Color Analysis**: Hue, saturation, value (HSV) metrics
- **📐 Shape Analysis**: Curvature, straightness, orientation
- **📍 Spatial Information**: Centroid coordinates, bounding box

## 🎓 Training

### Training Your Own Model

1. **Prepare your dataset** in YOLO format
2. **Update configuration** in `configs/training_config.yaml`
3. **Run training**:

```bash
# Local training
python3 scripts/train_yolo12.py

# Google Colab training
# Use notebooks/yolo12_colab_training.ipynb
```

### Training Configuration

Key parameters in `configs/yolo12_training_config.yaml`:

```yaml
model: yolo12s
epochs: 100
batch_size: 8
imgsz: 640
device: cpu  # or cuda for GPU
```

## 🔍 Inference

### Enhanced Trait Extractor

The main inference class combines YOLO12 detection with SAM2 segmentation:

```python
from src.inference.enhanced_trait_extractor import EnhancedTraitExtractor

# Initialize extractor
extractor = EnhancedTraitExtractor("models/yolo12/weights/best.pt")

# Process image
results = extractor.process_image("path/to/image.jpg", "output_dir")
```

### Output Format

Results are saved as JSON with:

```json
{
  "image_path": "path/to/image.jpg",
  "detections": {
    "cucumber": {...},
    "ruler": {...},
    "label": {...},
    "color_chart": {...}
  },
  "cucumber_traits": {
    "length_mm": 574.82,
    "width_mm": 148.48,
    "curvature": 0.01,
    "color_metrics": {...}
  },
  "segmentation_model": "sam2",
  "model_info": {...}
}
```

## 📊 Results

### Performance Metrics

- **YOLO12 Detection**: 95%+ accuracy on annotated test set
- **SAM2 Segmentation**: High-quality masks for precise trait extraction
- **Processing Speed**: ~110ms per image on CPU
- **Trait Accuracy**: Calibrated measurements with ruler reference

### Sample Output

The pipeline generates:
- **JSON Results**: Detailed trait measurements and metadata
- **Visualization Images**: Annotated images with detections and segmentation
- **Calibration Reports**: Measurement validation and quality metrics

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLO12**: Ultralytics for the latest YOLO model
- **SAM2**: Meta AI for the Segment Anything Model 2
- **Roboflow**: Dataset annotation and management tools
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/grover1012/Cucumber-HTP/issues)
- **Discussions**: [GitHub Discussions](https://github.com/grover1012/Cucumber-HTP/discussions)
- **Repository**: [https://github.com/grover1012/Cucumber-HTP](https://github.com/grover1012/Cucumber-HTP)

## 🔬 Research Applications

This pipeline is designed for:

- **Agricultural Research**: Cucumber phenotyping studies
- **Plant Breeding**: Trait correlation analysis
- **Quality Assessment**: Automated cucumber grading
- **Data Collection**: Systematic trait measurement
- **Educational Purposes**: Computer vision and agriculture learning

---

**Made with ❤️ for agricultural research and computer vision enthusiasts**
