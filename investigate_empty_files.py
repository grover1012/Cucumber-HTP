#!/usr/bin/env python3
"""
Investigate why some SAM2 annotation files have 0 annotations
"""

import json
from pathlib import Path

def investigate_empty_files():
    print("🔍 INVESTIGATING EMPTY ANNOTATION FILES")
    print("=" * 60)
    
    train_dir = Path("data/sam2_annotations/train")
    
    if not train_dir.exists():
        print("❌ Training directory not found!")
        return
    
    # Get all JSON files
    json_files = list(train_dir.glob("*.json"))
    print(f"📸 Found {len(json_files)} JSON annotation files")
    
    # Analyze each file
    empty_files = []
    files_with_annotations = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', [])
            image_info = data.get('image', {})
            image_name = image_info.get('file_name', 'unknown')
            
            if len(annotations) == 0:
                empty_files.append((json_file, data, image_name))
            else:
                files_with_annotations.append((json_file, len(annotations), image_name))
                
        except Exception as e:
            print(f"❌ Error reading {json_file.name}: {e}")
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"  • Files with annotations: {len(files_with_annotations)}")
    print(f"  • Files without annotations: {len(empty_files)}")
    print(f"  • Total annotations: {sum(count for _, count, _ in files_with_annotations)}")
    
    # Check empty files
    print(f"\n🔍 EMPTY FILES ANALYSIS:")
    print("-" * 40)
    
    for i, (json_file, data, image_name) in enumerate(empty_files[:10]):  # Show first 10
        print(f"\n📁 Empty file {i+1}: {json_file.name}")
        
        # Check file structure
        print(f"  📋 File structure:")
        for key in data.keys():
            if key == 'annotations':
                print(f"    • {key}: {len(data[key])} items")
            elif key == 'image':
                img_info = data[key]
                print(f"    • {key}: {img_info.get('file_name', 'unknown')} ({img_info.get('width', 0)}x{img_info.get('height', 0)})")
            else:
                print(f"    • {key}: {data[key]}")
        
        # Check if image file exists
        image_file = train_dir / image_name
        if image_file.exists():
            print(f"  ✅ Image file exists: {image_name}")
        else:
            print(f"  ❌ Image file missing: {image_name}")
    
    # Check files with annotations
    print(f"\n✅ FILES WITH ANNOTATIONS:")
    print("-" * 40)
    
    for i, (json_file, count, image_name) in enumerate(files_with_annotations[:10]):  # Show first 10
        print(f"  {i+1}. {json_file.name}: {count} annotations")
    
    # Look for patterns
    print(f"\n🔍 PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Check if empty files follow a pattern
    empty_patterns = {}
    for json_file, _, _ in empty_files:
        # Extract pattern from filename
        if 'AM' in json_file.name:
            parts = json_file.name.split('_')
            if len(parts) >= 2:
                pattern = f"{parts[0]}_{parts[1]}"
                empty_patterns[pattern] = empty_patterns.get(pattern, 0) + 1
    
    if empty_patterns:
        print(f"  📊 Empty files by pattern:")
        for pattern, count in sorted(empty_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"    • {pattern}: {count} empty files")
    
    # Check if it's a Roboflow export issue
    print(f"\n🎯 POTENTIAL ISSUES:")
    print("-" * 40)
    
    if len(empty_files) > len(json_files) * 0.4:  # More than 40% empty
        print("⚠️  HIGH PERCENTAGE OF EMPTY FILES - Possible issues:")
        print("   • Roboflow export problem")
        print("   • Incomplete annotation process")
        print("   • Dataset split issue")
        print("   • Some images may not have been annotated")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if len(empty_files) > 0:
        print("   • Check Roboflow project for annotation status")
        print("   • Verify all images were properly annotated")
        print("   • Consider re-exporting from Roboflow")
        print("   • Filter out empty files before training")
    
    # Check if we have enough good data
    total_annotations = sum(count for _, count, _ in files_with_annotations)
    if total_annotations > 500:
        print("   ✅ Sufficient annotations for training despite empty files")
    else:
        print("   ⚠️  May need more annotated data for good training")

if __name__ == "__main__":
    investigate_empty_files()
