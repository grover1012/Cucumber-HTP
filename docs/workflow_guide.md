# Cucumber Trait Extraction - Complete Workflow Guide

This guide provides a step-by-step workflow for extracting phenotypic traits from cucumber images using YOLO-based object detection and computer vision techniques.

## Table of Contents

1. [Project Setup](#project-setup)
2. [Data Preparation](#data-preparation)
3. [Annotation Guidelines](#annotation-guidelines)
4. [Model Training](#model-training)
5. [Inference and Trait Extraction](#inference-and-trait-extraction)
6. [Results Analysis](#results-analysis)
7. [Troubleshooting](#troubleshooting)

## Project Setup

### 1.1 Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd cucumber_HTP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 1.2 Directory Structure

```
cucumber_HTP/
├── data/
│   ├── raw_images/          # Your cucumber images
│   ├── annotations/          # YOLO format annotations
│   └── processed/           # Preprocessed images
├── models/                  # Trained YOLO models
├── configs/                 # Configuration files
├── src/                     # Source code
├── scripts/                 # Training and inference scripts
├── examples/                # Example usage
├── results/                 # Output results
└── docs/                    # Documentation
```

## Data Preparation

### 2.1 Image Requirements

- **Format**: JPG, JPEG, PNG, BMP, TIFF
- **Resolution**: Minimum 640x640, recommended 1920x1080 or higher
- **Content**: Each image should contain:
  - One or more cucumber fruits
  - A ruler for size reference
  - A paper label with accession ID
  - A color chart for color calibration

### 2.2 Image Organization

```bash
# Place your images in the raw_images directory
cp /path/to/your/cucumber/images/* data/raw_images/
```

### 2.3 Image Quality Guidelines

- Ensure good lighting conditions
- Avoid shadows and reflections
- Keep ruler markings clearly visible
- Ensure label text is readable
- Position color chart in well-lit area

## Annotation Guidelines

### 3.1 Annotation Tools

**Recommended Tools:**
- **CVAT** (Computer Vision Annotation Tool): Free, web-based, excellent YOLO export
- **Roboflow**: Cloud-based platform with advanced features
- **Label Studio**: Open-source annotation platform

### 3.2 Class Definitions

| Class ID | Class Name | Description | Annotation Type |
|----------|------------|-------------|-----------------|
| 0 | cucumber | Main fruit object | Segmentation mask |
| 1 | ruler | Measurement reference | Bounding box |
| 2 | label | Paper label with accession ID | Bounding box |
| 3 | color_chart | Color calibration reference | Bounding box |

### 3.3 Annotation Guidelines

#### Cucumber (Class 0)
- Draw segmentation mask around entire fruit
- Include from tip to tip
- Avoid including stems/leaves unless part of fruit
- Ensure tight boundary around fruit

#### Ruler (Class 1)
- Draw bounding box around entire ruler
- Include all visible measurement markings
- Ensure ruler is fully visible

#### Label (Class 2)
- Draw bounding box around paper label
- Include entire label area
- Ensure text is clearly readable

#### Color Chart (Class 3)
- Draw bounding box around entire color chart
- Include all color patches
- Ensure chart is well-lit and visible

### 3.4 Export Format

Export annotations in YOLO format:
- One `.txt` file per image
- Each line: `class_id x_center y_center width height`
- Coordinates normalized to [0, 1]
- For segmentation: Use polygon format if available

## Model Training

### 4.1 Dataset Preparation

```bash
# Prepare dataset for training
python scripts/train_yolo.py \
    --config configs/training_config.yaml \
    --raw-images data/raw_images \
    --annotations data/annotations \
    --prepare-only
```

### 4.2 Training Configuration

Edit `configs/training_config.yaml`:
- **Model**: Choose YOLOv8 architecture (nano to extra-large)
- **Epochs**: Start with 100, adjust based on performance
- **Batch size**: 16 for most GPUs, adjust based on memory
- **Image size**: 640x640 recommended for balance of speed/accuracy

### 4.3 Start Training

```bash
# Train the model
python scripts/train_yolo.py \
    --config configs/training_config.yaml \
    --train-only
```

### 4.4 Training Monitoring

- Monitor training metrics in `models/cucumber_traits/`
- Check for overfitting (validation loss increasing)
- Use early stopping if configured
- Export model when satisfied with performance

## Inference and Trait Extraction

### 5.1 Single Image Processing

```bash
# Process a single image
python scripts/extract_traits.py \
    --model models/best.pt \
    --image data/raw_images/sample.jpg \
    --output-dir results/single_image
```

### 5.2 Batch Processing

```bash
# Process multiple images
python scripts/extract_traits.py \
    --model models/best.pt \
    --image-dir data/raw_images \
    --output-dir results/batch_processing
```

### 5.3 Python API Usage

```python
from src.inference.trait_extractor import CucumberTraitExtractor

# Initialize extractor
extractor = CucumberTraitExtractor("models/best.pt", confidence_threshold=0.5)

# Process single image
results = extractor.process_image("path/to/image.jpg", "output/dir")

# Process multiple images
batch_results = extractor.batch_process(["image1.jpg", "image2.jpg"], "output/dir")
```

## Results Analysis

### 6.1 Output Files

For each processed image, the pipeline generates:
- `*_normalized.jpg`: Color-normalized image
- `*_visualization.jpg`: Image with detection overlays
- `*_results.json`: Complete extraction results
- `*_calibration.txt`: Calibration report
- `*_ocr.txt`: OCR extraction report

### 6.2 Results Structure

```json
{
  "image_path": "path/to/image.jpg",
  "detections": {
    "cucumber": {...},
    "ruler": {...},
    "label": {...},
    "color_chart": {...}
  },
  "calibration": {
    "ruler": {...},
    "color_chart": {...},
    "illumination": {...}
  },
  "cucumber_traits": {
    "length": 150.5,
    "width": 25.3,
    "aspect_ratio": 5.95,
    "area_mm2": 3812.6,
    "volume_cm3": 9.6
  },
  "accession_id": {
    "accession_id": "CU001",
    "confidence": 0.85,
    "is_valid": true
  }
}
```

### 6.3 Quality Metrics

- **Detection Confidence**: >0.7 recommended
- **Calibration Confidence**: >0.6 recommended
- **OCR Confidence**: >0.6 recommended
- **Measurement Validation**: All checks should pass

## Troubleshooting

### 7.1 Common Issues

#### Low Detection Confidence
- Check image quality and lighting
- Verify annotation quality
- Consider retraining with more data
- Adjust confidence threshold

#### Poor Calibration
- Ensure ruler is clearly visible
- Check ruler markings are distinct
- Verify ruler is not occluded
- Consider manual calibration

#### OCR Failures
- Ensure label is well-lit
- Check text contrast
- Verify label is not blurry
- Try different preprocessing settings

#### Memory Issues
- Reduce batch size
- Use smaller model architecture
- Process images individually
- Check GPU memory usage

### 7.2 Performance Optimization

- Use GPU acceleration when available
- Batch process images for efficiency
- Use appropriate model size for your needs
- Optimize image resolution based on requirements

### 7.3 Data Augmentation

If training performance is poor:
- Increase training data
- Apply data augmentation
- Use transfer learning
- Balance class distribution

## Advanced Usage

### 8.1 Custom Trait Extraction

```python
from src.utils.trait_extraction import extract_all_traits

# Extract specific traits
traits = extract_all_traits(mask, image, pixel_to_mm_ratio)
```

### 8.2 Custom Calibration

```python
from src.utils.calibration import calibrate_with_known_object

# Use known object dimensions
calibration = calibrate_with_known_object(ruler_mask, image, known_length_mm=150.0)
```

### 8.3 Batch Processing with Custom Logic

```python
# Custom batch processing
for image_path in image_paths:
    results = extractor.process_image(image_path)
    # Custom post-processing
    custom_analysis(results)
```

## Best Practices

1. **Data Quality**: Ensure high-quality images and annotations
2. **Regular Validation**: Monitor model performance regularly
3. **Documentation**: Keep track of processing parameters and results
4. **Backup**: Regularly backup trained models and results
5. **Version Control**: Use version control for code and configurations

## Support and Resources

- **Documentation**: Check `docs/` directory for detailed guides
- **Examples**: See `examples/` directory for usage examples
- **Issues**: Report bugs and request features through issue tracker
- **Community**: Join discussions and share experiences

---

This workflow guide covers the complete pipeline from raw images to trait extraction. Follow these steps systematically to achieve optimal results with your cucumber trait extraction project.
