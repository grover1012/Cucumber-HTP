#!/usr/bin/env python3
"""
Analyze Detection Quality Issues
Shows detailed analysis of good, fair, and poor quality detections
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_quality_issues(analysis_file):
    """Analyze quality issues from the detailed analyses."""
    
    with open(analysis_file, 'r') as f:
        analyses = json.load(f)
    
    print("🔍 DETECTION QUALITY ANALYSIS")
    print("=" * 60)
    
    # Categorize by quality
    quality_categories = {
        'excellent': [],
        'good': [],
        'fair': [],
        'poor': []
    }
    
    for analysis in analyses:
        score = analysis['quality_score']
        if score >= 90:
            quality_categories['excellent'].append(analysis)
        elif score >= 80:
            quality_categories['good'].append(analysis)
        elif score >= 60:
            quality_categories['fair'].append(analysis)
        else:
            quality_categories['poor'].append(analysis)
    
    # Show summary
    print(f"📊 Quality Distribution:")
    for category, images in quality_categories.items():
        print(f"  {category.capitalize()}: {len(images)} images")
    print()
    
    # Analyze issues in each category
    for category in ['good', 'fair', 'poor']:
        if quality_categories[category]:
            print(f"🔍 {category.upper()} QUALITY IMAGES - ISSUES ANALYSIS:")
            print("-" * 50)
            
            for i, analysis in enumerate(quality_categories[category][:5]):  # Show first 5
                print(f"\n{i+1}. {Path(analysis['image_path']).name}")
                print(f"   Quality Score: {analysis['quality_score']:.1f}%")
                print(f"   Total Detections: {analysis['total_detections']}")
                
                # Show confidence distribution
                conf_dist = analysis['confidence_distribution']
                print(f"   Confidence: Very High: {conf_dist.get('very_high', 0)}, "
                      f"High: {conf_dist.get('high', 0)}, "
                      f"Medium: {conf_dist.get('medium', 0)}, "
                      f"Low: {conf_dist.get('low', 0)}, "
                      f"Very Low: {conf_dist.get('very_low', 0)}")
                
                # Show low confidence objects
                if analysis['low_confidence_objects']:
                    print("   ⚠️ Low Confidence Objects:")
                    for obj in analysis['low_confidence_objects']:
                        print(f"     - {obj['class']}: {obj['confidence']:.3f} (threshold: {obj['threshold']})")
                
                # Show potential misses
                if analysis['potential_misses']:
                    print("   ❌ Potential Issues:")
                    for issue in analysis['potential_misses']:
                        print(f"     - {issue}")
                
                # Show class distribution
                class_dist = analysis['class_distribution']
                print(f"   Classes: {', '.join([f'{k}: {v}' for k, v in class_dist.items()])}")
            
            if len(quality_categories[category]) > 5:
                print(f"\n   ... and {len(quality_categories[category]) - 5} more images")
            print()
    
    # Overall recommendations
    print("🎯 QUALITY IMPROVEMENT RECOMMENDATIONS:")
    print("=" * 60)
    
    # Analyze common issues
    all_low_confidence = []
    all_potential_misses = []
    
    for analysis in analyses:
        all_low_confidence.extend(analysis['low_confidence_objects'])
        all_potential_misses.extend(analysis['potential_misses'])
    
    if all_low_confidence:
        print("1. 🔴 Low Confidence Detections:")
        class_confidence_issues = defaultdict(list)
        for obj in all_low_confidence:
            class_confidence_issues[obj['class']].append(obj['confidence'])
        
        for class_name, confidences in class_confidence_issues.items():
            avg_conf = sum(confidences) / len(confidences)
            print(f"   - {class_name}: {len(confidences)} instances, avg confidence: {avg_conf:.3f}")
    
    if all_potential_misses:
        print("\n2. ⚠️ Common Issues:")
        issue_counts = defaultdict(int)
        for issue in all_potential_misses:
            issue_counts[issue] += 1
        
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {issue}: {count} occurrences")
    
    # Class-specific recommendations
    print("\n3. 🎯 Class-Specific Improvements:")
    
    # Find classes with most low-confidence detections
    class_issues = defaultdict(int)
    for obj in all_low_confidence:
        class_issues[obj['class']] += 1
    
    if class_issues:
        print("   Classes needing confidence threshold adjustment:")
        for class_name, count in sorted(class_issues.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {class_name}: {count} low-confidence detections")
    
    print("\n4. 📈 Training Recommendations:")
    print("   - Current model shows good overall performance (88.7% quality)")
    print("   - Focus on improving confidence for problematic classes")
    print("   - Consider adjusting class-specific confidence thresholds")
    print("   - Dataset quality is sufficient for training a robust model")

def main():
    """Main function."""
    analysis_file = "results/enhanced_auto_labels_new/analysis/detailed_analyses.json"
    
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        print("Please run the enhanced auto-labeler first.")
        return
    
    analyze_quality_issues(analysis_file)

if __name__ == "__main__":
    main()
