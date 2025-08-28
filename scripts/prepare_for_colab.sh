#!/bin/bash
# Prepare Clean Dataset for Colab Upload
# This script creates a zip file ready for Colab

echo "📦 Preparing clean dataset for Colab upload..."

# Create zip file
cd data
zip -r clean_dataset_for_colab.zip clean_dataset/
cd ..

echo "✅ Dataset prepared: data/clean_dataset_for_colab.zip"
echo "📤 Upload this file to Google Colab!"
echo ""
echo "📋 Next steps:"
echo "1. Go to Google Colab: https://colab.research.google.com/"
echo "2. Create new notebook"
echo "3. Upload clean_dataset_for_colab.zip"
echo "4. Run the training cells"
echo "5. Download your trained model"
