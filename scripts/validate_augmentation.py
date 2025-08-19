"""Quick script to preview augmentation pipeline."""

from pathlib import Path
import numpy as np
import yaml

try:
    import albumentations as A
    import cv2
except Exception as e:  # pragma: no cover - best effort if deps missing
    print(f"Albumentations not available: {e}")
    A = None


def build_pipeline(cfg):
    """Build Albumentations pipeline from config."""
    if A is None:
        return None
    aug_cfg = cfg.get("augmentation", {})
    transforms = []
    if aug_cfg.get("hsv_h") or aug_cfg.get("hsv_s") or aug_cfg.get("hsv_v"):
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=int(aug_cfg.get("hsv_h", 0) * 255),
                sat_shift_limit=int(aug_cfg.get("hsv_s", 0) * 255),
                val_shift_limit=int(aug_cfg.get("hsv_v", 0) * 255),
                p=1.0,
            )
        )
    if aug_cfg.get("flipud", 0) > 0:
        transforms.append(A.VerticalFlip(p=aug_cfg["flipud"]))
    if aug_cfg.get("fliplr", 0) > 0:
        transforms.append(A.HorizontalFlip(p=aug_cfg["fliplr"]))
    return A.Compose(transforms) if transforms else None


def main():
    cfg = yaml.safe_load(Path("configs/yolo12_training_config.yaml").read_text())
    pipeline = build_pipeline(cfg)
    if pipeline is None:
        print("Augmentation pipeline could not be created.")
        return

    size = cfg["training"].get("imgsz", 640)
    image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    augmented = pipeline(image=image)["image"]
    out_path = Path("augmented_sample.jpg")
    cv2.imwrite(str(out_path), augmented)
    print(f"Saved augmented sample to {out_path}")


if __name__ == "__main__":
    main()

