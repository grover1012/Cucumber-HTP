"""
Cucumber High-Throughput Phenotyping (HTP) Pipeline
Inspired by TomatoScanner's approach for robust phenotyping.

This package provides a complete pipeline for cucumber trait extraction using:
- EdgeBoost augmentation for better edge detection
- EdgeLoss for precise segmentation boundaries
- Pose normalization for consistent measurements
- Ruler-based scaling for pixel-to-cm conversion
- Comprehensive trait extraction
- OCR integration for accession labels
"""

__version__ = "1.0.0"
__author__ = "Cucumber HTP Team"
__description__ = "Robust cucumber phenotyping pipeline inspired by TomatoScanner"

# Core pipeline components
from .edge_aug import EdgeBoost, edgeboost, edgeboost_transform, create_edgeboost_pipeline
from .edge_losses import EdgeLoss, edge_loss_simple, compute_edge_metrics
from .pose import rotate_to_major_axis, normalize_multiple_cucumbers, compute_pose_metrics
from .scale_ruler import RulerDetector, find_ppcm_simple, find_ppcm_robust
from .traits import CucumberTraitExtractor, extract_traits_simple, extract_traits_advanced
from .ocr_label import AccessionLabelReader, read_label_simple, read_label_robust

# Training and inference
from .train_seg import train, create_edge_training_config
from .infer_seg import CucumberPhenotypingPipeline, run_folder_pipeline

# Main pipeline driver
from .run_pipeline import main as run_pipeline

__all__ = [
    # Core components
    'EdgeBoost',
    'edgeboost',
    'edgeboost_transform',
    'create_edgeboost_pipeline',
    'EdgeLoss',
    'edge_loss_simple',
    'compute_edge_metrics',
    'rotate_to_major_axis',
    'normalize_multiple_cucumbers',
    'compute_pose_metrics',
    'RulerDetector',
    'find_ppcm_simple',
    'find_ppcm_robust',
    'CucumberTraitExtractor',
    'extract_traits_simple',
    'extract_traits_advanced',
    'AccessionLabelReader',
    'read_label_simple',
    'read_label_robust',
    
    # Training and inference
    'train',
    'create_edge_training_config',
    'CucumberPhenotypingPipeline',
    'run_folder_pipeline',
    
    # Main driver
    'run_pipeline',
]
