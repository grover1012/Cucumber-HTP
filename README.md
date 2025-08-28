# 🥒 Cucumber High-Throughput Phenotyping (HTP) Pipeline

**Inspired by [TomatoScanner](https://github.com/AlexTraveling/TomatoScanner) - A robust, publication-grade pipeline for cucumber trait extraction using only RGB images.**

## 🚀 **What You'll Build**

This pipeline transforms your cucumber images into precise phenotypic measurements:

* **`detector`**: YOLO-seg model with EdgeBoost + EdgeLoss for edge-accurate segmentation
* **`pose`**: Auto-rotate each fruit to canonical horizontal orientation using major axis
* **`scale`**: Detect ruler → compute pixels-per-cm (no monocular depth needed)
* **`traits`**: Length, max width, aspect ratio, area, curvature, color analysis
* **`ocr`**: Read accession labels and link traits to genetic data
* **`export`**: Per-fruit CSV rows for downstream statistical analysis

## 🏗️ **Architecture Overview**

```
Input Image → EdgeBoost → YOLO-Seg → Pose Normalization → Trait Extraction → Scale Conversion → OCR → CSV Export
     ↓              ↓         ↓            ↓              ↓              ↓         ↓      ↓
  RGB Image   Contrast+    Masks      Horizontal    Length/Width   cm units   Labels  Results
              Acutance    (EdgeLoss)   Orientation     Area/Curv
```

## 📋 **Key Features**

### **🎯 Edge-Aware Segmentation**
- **EdgeBoost**: CLAHE + Unsharp masking for better edge detection
- **EdgeLoss**: Sobel gradient-based loss for precise mask boundaries
- **Inspired by TomatoScanner's EdgeAttention + EdgeLoss modules**

### **🔄 Pose Normalization**
- Automatic rotation to horizontal orientation using principal component analysis
- Ensures consistent measurements regardless of fruit orientation
- Handles multiple cucumbers per image

### **📏 Ruler-Based Scaling**
- **No depth camera required** - uses your ruler for pixel-to-cm conversion
- Robust tick detection with Hough line transforms
- Optional OCR validation of ruler numbers

### **📊 Comprehensive Trait Extraction**
- **Basic**: Length, width, area, aspect ratio, perimeter
- **Advanced**: Curvature analysis, fractal dimension, convexity defects
- **Color**: RGB analysis, color chart normalization

### **🏷️ OCR Integration**
- **EasyOCR + Tesseract** for robust text recognition
- Automatic accession number extraction using regex patterns
- Links phenotypic data to genetic identifiers

## 🚀 **Quick Start**

### **1. Environment Setup**

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Dataset Preparation**

Export your Roboflow project in **YOLOv8 Segmentation** format:

```bash
# Your data structure should look like:
data/
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels
  data.yaml
```

### **3. Training with EdgeBoost + EdgeLoss**

```bash
# Train edge-aware segmentation model
python src/run_pipeline.py --stage train \
  --model yolov8m-seg.pt \
  --epochs 150 \
  --imgsz 1024 \
  --batch 8 \
  --edge-lambda 0.3 \
  --edgeboost-prob 0.5
```

### **4. Run Complete Phenotyping Pipeline**

```bash
# Process test images
python src/run_pipeline.py --stage infer \
  --model outputs/models/yolov8m-seg-edge/weights/best.pt \
  --img-dir data/test/images \
  --output-csv outputs/tables/phenotyping_results.csv
```

## 📁 **Repository Structure**

```
cucumber-htp/
├── src/                          # Core pipeline modules
│   ├── edge_aug.py              # EdgeBoost augmentation
│   ├── edge_losses.py           # EdgeLoss implementation
│   ├── pose.py                  # Pose normalization
│   ├── scale_ruler.py           # Ruler detection & scaling
│   ├── traits.py                # Trait extraction
│   ├── ocr_label.py             # OCR for accession labels
│   ├── train_seg.py             # Training with EdgeLoss
│   ├── infer_seg.py             # Complete inference pipeline
│   └── run_pipeline.py          # Main pipeline driver
├── data/                         # Your dataset
├── outputs/                      # Results and models
│   ├── models/                  # Trained models
│   ├── predictions/             # Detection results
│   ├── visualizations/          # Output images
│   └── tables/                  # CSV/JSON results
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

## 🔧 **Core Components**

### **EdgeBoost Augmentation (`src/edge_aug.py`)**
```python
from src.edge_aug import edgeboost

# Apply edge enhancement
enhanced_img = edgeboost(img, lambda_c=1.5, lambda_a=1.6)
```

**Features:**
- **CLAHE**: Adaptive histogram equalization for contrast
- **Unsharp Masking**: Edge sharpening for better segmentation
- **Albumentations integration**: Seamless pipeline integration

### **EdgeLoss (`src/edge_losses.py`)**
```python
from src.edge_losses import EdgeLoss

# Create edge-aware loss
edge_loss = EdgeLoss(lambda_edge=0.3)
loss = edge_loss(pred_masks, gt_masks)
```

**Features:**
- **Sobel gradients**: Edge-aware loss computation
- **L1 penalty**: Differences in edge gradients
- **Configurable weights**: Balance segmentation vs. edge accuracy

### **Pose Normalization (`src/pose.py`)**
```python
from src.pose import rotate_to_major_axis

# Normalize cucumber orientation
rotated_img, rotated_mask, angle = rotate_to_major_axis(img, mask)
```

**Features:**
- **Principal Component Analysis**: Major axis detection
- **Automatic rotation**: Horizontal orientation
- **Multiple cucumber support**: Batch processing

### **Ruler Detection (`src/scale_ruler.py`)**
```python
from src.scale_ruler import find_ppcm_simple

# Detect scale from ruler
ppcm = find_ppcm_simple(img, expected_tick_distance_cm=1.0)
```

**Features:**
- **Hough line detection**: Robust ruler tick identification
- **OCR validation**: Optional number reading
- **Configurable tick distances**: Adapt to your ruler

### **Trait Extraction (`src/traits.py`)**
```python
from src.traits import CucumberTraitExtractor

extractor = CucumberTraitExtractor()
traits = extractor.extract_advanced_traits(mask)
```

**Features:**
- **20+ morphological traits**: Comprehensive phenotyping
- **Curvature analysis**: Shape complexity metrics
- **Fractal dimension**: Self-similarity analysis

### **OCR Integration (`src/ocr_label.py`)**
```python
from src.ocr_label import AccessionLabelReader

reader = AccessionLabelReader()
accession = reader.read_accession_label(img)
```

**Features:**
- **Dual OCR engines**: EasyOCR + Tesseract
- **Accession patterns**: Regex-based extraction
- **Confidence scoring**: Quality assessment

## 📊 **Training Configuration**

### **Edge-Aware Training Parameters**

```yaml
# Key parameters for edge-aware training
edge_lambda: 0.3              # Weight for edge loss
edgeboost_prob: 0.5           # Probability of EdgeBoost
lambda_c: 1.5                 # Contrast enhancement
lambda_a: 1.6                 # Acutance enhancement
imgsz: 1024                   # High resolution for edges
batch: 8                      # Optimized for memory
epochs: 150                   # Sufficient convergence
```

### **Model Selection Guide**

| Model | Parameters | Memory | Speed | Quality | Use Case |
|-------|------------|---------|-------|---------|----------|
| `yolov8n-seg.pt` | 3.2M | Low | Fast | Good | Quick prototyping |
| `yolov8s-seg.pt` | 11.2M | Medium | Medium | Better | Production baseline |
| `yolov8m-seg.pt` | 25.9M | High | Slow | Best | Publication quality |
| `yolov8l-seg.pt` | 43.7M | Very High | Slowest | Excellent | Maximum accuracy |

## 🔍 **Inference Pipeline**

### **Complete Processing Flow**

1. **Image Loading** → RGB image input
2. **Ruler Detection** → Scale computation (pixels/cm)
3. **Cucumber Detection** → YOLO segmentation
4. **Pose Normalization** → Horizontal orientation
5. **Trait Extraction** → 20+ morphological features
6. **Scale Conversion** → Pixel → cm conversion
7. **OCR Reading** → Accession label extraction
8. **Results Export** → CSV + JSON + visualizations

### **Output Formats**

**CSV Results:**
```csv
image_name,cucumber_id,accession_label,cm_length_cm,cm_width_cm,aspect_ratio,area_cm2
IMG_001.jpg,0,ABC123,14.2,3.2,4.44,45.44
IMG_001.jpg,1,DEF456,16.8,3.5,4.80,58.80
```

**JSON Results:**
```json
{
  "image_name": "IMG_001.jpg",
  "cucumbers": [
    {
      "cucumber_id": 0,
      "traits_cm": {"length_cm": 14.2, "width_cm": 3.2},
      "pose_metadata": {"rotation_angle": 15.3}
    }
  ]
}
```

## 📈 **Performance Validation**

### **Manual Validation Template**

```bash
# Create validation template
python src/run_pipeline.py --stage template
```

**Fill in manual measurements:**
```csv
image_name,cucumber_id,length_cm_gt,width_cm_gt,notes
IMG_001.jpg,0,14.2,3.2,Manual caliper measurement
```

### **Relative Error Calculation**

```python
import pandas as pd

# Load results
p = pd.read_csv("outputs/tables/phenotyping_results.csv")
g = pd.read_csv("manual_validation.csv")

# Merge and compute relative error
m = p.merge(g, on=["image_name", "cucumber_id"])
m["length_RE_%"] = 100 * (m["cm_length_cm"] - m["length_cm_gt"]) / m["length_cm_gt"]
m["width_RE_%"] = 100 * (m["cm_width_cm"] - m["width_cm_gt"]) / m["width_cm_gt"]

print(m[["length_RE_%", "width_RE_%"]].describe())
```

**Target Performance:**
- **Length/Width**: ≤5-7% median relative error
- **Area**: ≤10-15% median relative error
- **Edge accuracy**: ≤3% edge error (TomatoScanner benchmark)

## 🎯 **Advanced Usage**

### **Custom EdgeBoost Parameters**

```python
from src.edge_aug import create_edgeboost_pipeline

# Create custom augmentation pipeline
pipeline = create_edgeboost_pipeline(
    lambda_c=2.0,      # Higher contrast
    lambda_a=2.0,      # Higher sharpness
    p=0.7              # Higher probability
)
```

### **Batch Processing with Custom Scale**

```python
from src.infer_seg import CucumberPhenotypingPipeline

# Initialize with custom parameters
pipeline = CucumberPhenotypingPipeline(
    model_path="path/to/model.pt",
    ruler_tick_distance_cm=0.5,  # 0.5cm ticks
    confidence_threshold=0.3,     # Lower threshold
    save_visualizations=True
)

# Process folder
results = pipeline.process_folder("path/to/images")
```

### **Integration with Existing Workflows**

```python
# Use individual components
from src.traits import extract_traits_advanced
from src.pose import rotate_to_major_axis

# Process custom masks
traits = extract_traits_advanced(your_mask)
rotated_img, rotated_mask, angle = rotate_to_major_axis(img, your_mask)
```

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

| Issue | Cause | Solution |
|-------|-------|----------|
| **EdgeLoss not working** | Ultralytics version mismatch | Update to ultralytics==8.3.34 |
| **Scale detection fails** | Ruler not visible/clear | Ensure ruler is vertical and well-lit |
| **OCR accuracy low** | Poor image quality | Use EdgeBoost preprocessing |
| **Memory errors** | Batch size too large | Reduce batch size or use smaller model |
| **Training divergence** | Edge loss weight too high | Reduce `edge_lambda` to 0.1-0.2 |

### **Debug Mode**

```bash
# Enable verbose logging
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -u src/run_pipeline.py --stage infer --img-dir data/test/images
```

## 📚 **References & Inspiration**

- **[TomatoScanner Paper](https://arxiv.org/abs/2503.05568)**: EdgeAttention, EdgeLoss, EdgeBoost modules
- **[YOLOv8 Documentation](https://docs.ultralytics.com/)**: Model architecture and training
- **[Albumentations](https://albumentations.ai/)**: Advanced augmentation techniques
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)**: Robust text recognition

## 🤝 **Contributing**

This pipeline is designed to be modular and extensible:

1. **Add new traits** in `src/traits.py`
2. **Implement new augmentations** in `src/edge_aug.py`
3. **Create custom losses** in `src/edge_losses.py`
4. **Extend OCR patterns** in `src/ocr_label.py`

## 📄 **License**

This project is inspired by TomatoScanner's research and implements similar concepts for cucumber phenotyping. Please cite both this pipeline and the original TomatoScanner paper if used in research.

## 🎉 **Getting Help**

- **Check the examples** in each module's `__main__` section
- **Validate your dataset** using the built-in validation
- **Start with small models** (yolov8n-seg) for testing
- **Use the template stage** to create validation files

---

**🚀 Ready to build your robust cucumber phenotyping pipeline? Start with the Quick Start section above!**
