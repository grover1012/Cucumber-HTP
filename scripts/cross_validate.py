#!/usr/bin/env python3
"""K-fold cross-validation training using ``YOLOTrainer``.

This script splits the existing ``data/annotations`` dataset into *k*
folds and trains a model on each fold using the :class:`YOLOTrainer`.  The
results from all folds are summarised to help select the best configuration
before running a full training run.
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Dict

import yaml
import numpy as np

# Add src directory to path for local imports
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from training.yolo_trainer import YOLOTrainer  # noqa: E402  (import after path append)


def create_kfold_datasets(base_dir: Path, k: int, class_names: List[str], nc: int) -> List[Path]:
    """Create *k* fold directories from ``base_dir``.

    Parameters
    ----------
    base_dir:
        Path to the dataset root (e.g. ``data/annotations``).
    k:
        Number of folds to create.
    class_names:
        List of class names used for ``dataset.yaml`` creation.
    nc:
        Number of classes.

    Returns
    -------
    List[Path]
        Paths to the newly created fold directories.
    """

    all_images: List[Path] = []
    for split in ["train", "valid"]:
        all_images.extend((base_dir / split / "images").glob("*"))

    all_images = [img for img in all_images if img.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(all_images)

    folds = np.array_split(all_images, k)
    folds_root = base_dir / "folds"
    shutil.rmtree(folds_root, ignore_errors=True)
    fold_paths: List[Path] = []

    for i in range(k):
        fold_dir = folds_root / f"fold_{i+1}"
        train_img_dir = fold_dir / "train" / "images"
        train_lbl_dir = fold_dir / "train" / "labels"
        val_img_dir = fold_dir / "val" / "images"
        val_lbl_dir = fold_dir / "val" / "labels"

        for path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            path.mkdir(parents=True, exist_ok=True)

        val_images = folds[i]
        train_images = [img for j, fold in enumerate(folds) if j != i for img in fold]

        for img_path in train_images:
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            shutil.copy2(img_path, train_img_dir / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, train_lbl_dir / label_path.name)
            else:
                (train_lbl_dir / label_path.name).touch()

        for img_path in val_images:
            label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            shutil.copy2(img_path, val_img_dir / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, val_lbl_dir / label_path.name)
            else:
                (val_lbl_dir / label_path.name).touch()

        dataset_yaml = {
            "path": str(fold_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": nc,
            "names": class_names,
        }

        with open(fold_dir / "dataset.yaml", "w") as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        fold_paths.append(fold_dir)

    return fold_paths


def summarise_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute average metrics across folds."""

    summary: Dict[str, float] = {}
    if not fold_metrics:
        return summary

    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        summary[key] = float(np.mean(values))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold cross-validation for YOLO12 training")
    parser.add_argument("--config", default="configs/yolo12_training_config.yaml", help="Path to training configuration")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to use")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="n", help="YOLO12 model size")
    parser.add_argument("--device", default="auto", help="Device for training")
    parser.add_argument("--output-dir", default="models/cross_validation", help="Directory to save fold models")
    args = parser.parse_args()

    trainer = YOLOTrainer(args.config)
    trainer.config["training"]["device"] = args.device

    model_architecture = f"yolo12{args.model_size}"

    dataset_root = Path(trainer.config["dataset"]["path"]) / "annotations"
    fold_dirs = create_kfold_datasets(dataset_root, args.folds, trainer.class_names, trainer.num_classes)

    results: List[Dict[str, float]] = []

    for fold_idx, fold_dir in enumerate(fold_dirs, 1):
        print(f"\n=== Training fold {fold_idx}/{args.folds} ===")
        trainer.config["dataset"]["path"] = str(fold_dir)
        trainer.dataset_path = str(fold_dir)

        best_model = trainer.train_model(model_architecture=model_architecture, output_dir=os.path.join(args.output_dir, f"fold_{fold_idx}"), use_yolo12=True)
        metrics = trainer.validate_model(best_model)
        results.append(metrics)

    print("\n=== Cross-validation summary ===")
    for i, m in enumerate(results, 1):
        print(f"Fold {i}: mAP50={m['mAP50']:.4f}, mAP50-95={m['mAP50-95']:.4f}, precision={m['precision']:.4f}, recall={m['recall']:.4f}")

    summary = summarise_metrics(results)
    print("\nAverage metrics across folds:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    best_fold = max(range(len(results)), key=lambda i: results[i]["mAP50"])
    print(f"\nBest fold: {best_fold + 1} with mAP50 {results[best_fold]['mAP50']:.4f}")


if __name__ == "__main__":
    main()

