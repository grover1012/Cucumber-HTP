# 🚀 Google Colab YOLO12 Training Guide

## ⚡ **Why Google Colab?**

- **GPU Acceleration**: 10-20x faster than CPU training
- **Free Tier**: T4 GPU available for free
- **No Setup**: Everything pre-installed
- **Cloud Storage**: No local storage limits

## 📋 **Prerequisites**

1. **Google Account** (free)
2. **Your Cucumber Dataset** (ZIP file)
3. **Internet Connection**

## 🎯 **Step-by-Step Guide**

### **Step 1: Open Google Colab**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Create a new notebook

### **Step 2: Enable GPU**
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### **Step 3: Upload Your Dataset**
1. **Zip your data folder**:
   ```bash
   # On your local machine
   cd cucumber_HTP
   zip -r cucumber_dataset.zip data/annotations/
   ```

2. **Upload to Colab**:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

3. **Extract dataset**:
   ```python
   import zipfile
   with zipfile.ZipFile('cucumber_dataset.zip', 'r') as zip_ref:
       zip_ref.extractall('.')
   ```

### **Step 4: Install Dependencies**
```python
!pip install ultralytics -q
!pip install roboflow -q
```

### **Step 5: Create Dataset Configuration**
```python
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

print("✅ Dataset configuration created!")
```

### **Step 6: Start YOLO12 Training**
```python
from ultralytics import YOLO

# Load YOLO12s model
model = YOLO("yolo12s")

# Start training
results = model.train(
    data='dataset.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,  # Use GPU
    project='cucumber_traits_yolo12',
    name='exp_s',
    save=True,
    plots=True
)

print("🎉 Training completed!")
```

### **Step 7: Download Results**
```python
from google.colab import files

# Download best model
files.download('cucumber_traits_yolo12/exp_s/weights/best.pt')

# Download training results
!zip -r results.zip cucumber_traits_yolo12/exp_s/
files.download('results.zip')
```

## 🎯 **Recommended Training Configuration**

### **For Quick Results (YOLO12n)**
```python
model = YOLO("yolo12n")
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    device=0
)
# Expected time: 15-30 minutes
```

### **For Best Balance (YOLO12s) - RECOMMENDED**
```python
model = YOLO("yolo12s")
results = model.train(
    data='dataset.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    device=0
)
# Expected time: 30-60 minutes
```

### **For Maximum Accuracy (YOLO12m)**
```python
model = YOLO("yolo12m")
results = model.train(
    data='dataset.yaml',
    epochs=200,
    imgsz=832,
    batch=8,
    device=0
)
# Expected time: 1-2 hours
```

## 📊 **Expected Performance**

| Model | Training Time | mAP50 | Best For |
|-------|---------------|-------|-----------|
| **YOLO12n** | 15-30 min | 0.75-0.80 | Quick prototyping |
| **YOLO12s** | 30-60 min | 0.80-0.85 | **Production use** ⭐ |
| **YOLO12m** | 1-2 hours | 0.85-0.90 | High accuracy |

## 🔧 **Troubleshooting**

### **GPU Not Available**
- Check Runtime → Change runtime type → GPU
- Wait for GPU allocation (may take a few minutes)

### **Out of Memory**
- Reduce batch size: `batch=8`
- Use smaller model: `yolo12n`
- Reduce image size: `imgsz=512`

### **Dataset Not Found**
- Check file paths in `dataset.yaml`
- Ensure ZIP file extracted correctly
- Verify folder structure matches expected

### **Training Too Slow**
- Use smaller model (n or s)
- Reduce epochs
- Check GPU is active

## 📥 **After Training**

### **Files to Download**
1. **`best.pt`** - Your trained model
2. **`results.zip`** - Training plots and metrics

### **Local Usage**
```bash
# Place model in your local project
cp best.pt cucumber_HTP/models/

# Run inference
python3 scripts/extract_traits.py \
    --model models/best.pt \
    --image data/annotations/test/images/sample.jpg
```

## 🌟 **Pro Tips**

1. **Save Colab Notebook**: File → Save a copy in Drive
2. **Use Drive Storage**: Mount Google Drive for larger datasets
3. **Monitor GPU**: Watch GPU memory usage during training
4. **Save Checkpoints**: Download models every 50 epochs
5. **Use Pro/Pro+**: For longer training sessions and better GPUs

## 🎉 **Success Metrics**

Your training is successful when:
- **mAP50 > 0.7** (70% accuracy)
- **Training loss decreases** steadily
- **Validation metrics improve**
- **No overfitting** detected

---

**Ready to train?** Follow these steps and you'll have a trained YOLO12 model in under an hour! 🚀
