#!/bin/bash
# Production Training Script for Cucumber Detection
# Generated automatically

echo "🚀 Starting Production Training..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start training
python3 -m ultralytics train \
    data=/Users/kgrover2/Documents/cucumber_HTP/data/clean_dataset/data.yaml \
    model=yolo12s.pt \
    epochs=500 \
    patience=50 \
    batch=16 \
    imgsz=640 \
    save_period=25 \
    cache=True \
    device=0 \
    workers=8 \
    project=models/production \
    name=cucumber_traits_v2 \
    exist_ok=True \
    pretrained=True \
    optimizer=AdamW \
    verbose=True \
    seed=42 \
    cos_lr=True \
    amp=True \
    multi_scale=True \
    dropout=0.1 \
    half=True \
    augment=True \
    degrees=10.0 \
    translate=0.2 \
    scale=0.9 \
    shear=2.0 \
    perspective=0.001 \
    flipud=0.1 \
    fliplr=0.5 \
    mosaic=1.0 \
    mixup=0.3 \
    cutmix=0.3 \
    copy_paste=0.1 \
    auto_augment=randaugment \
    erasing=0.4 \
    lr0=0.001 \
    lrf=0.01 \
    warmup_epochs=5.0 \
    weight_decay=0.0005

echo "✅ Training completed!"
