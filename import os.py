import os
import random
import shutil

# === CONFIGURATION ===
raw_images_dir = "/Users/kgrover2/Downloads/Cucumber_HTP/images"  # change to your actual folder
output_dir = "/Users/kgrover2/Downloads/Cucumber_HTP/selected_for_annotation/"
num_samples = 250

# === MAKE SURE OUTPUT DIR EXISTS ===
os.makedirs(output_dir, exist_ok=True)

# === GET IMAGE FILES ===
image_files = [f for f in os.listdir(raw_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# === RANDOMLY SELECT ===
selected_images = random.sample(image_files, min(num_samples, len(image_files)))

# === COPY TO NEW FOLDER ===
for fname in selected_images:
    src = os.path.join(raw_images_dir, fname)
    dst = os.path.join(output_dir, fname)
    shutil.copy(src, dst)

print(f"✅ Copied {len(selected_images)} images to '{output_dir}'")
