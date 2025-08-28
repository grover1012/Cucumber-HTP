#!/usr/bin/env python3
"""
Segmentation Visualizer
- Runs YOLO detection
- Generates segmentation masks per detection using SAM2 if available, else a smart fallback
- Saves an overlay image with masks + per-object mask PNGs
"""

import os
import cv2
import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# Optional: SAM2 integration
try:
	from segment_anything_hq import sam_model_registry, SamPredictor  # hypothetical import path
	sam_available = True
except Exception:
	sam_available = False

CLASS_NAMES = [
	'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
	'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
]

COLORS = {
	'cucumber': (0, 255, 0),
	'slice': (255, 165, 0),
	'ruler': (255, 0, 0),
	'big_ruler': (255, 0, 255),
	'color_chart': (0, 0, 255),
	'cavity': (0, 255, 255),
	'hollow': (128, 128, 128),
	'label': (255, 255, 0),
	'objects': (128, 0, 128),
	'blue_dot': (255, 0, 128),
	'green_dot': (0, 128, 0),
	'red_dot': (128, 0, 0)
}

def load_sam(model_path: str):
	if not sam_available:
		return None
	try:
		# This is pseudocode; adapt to your SAM2 package
		sam = sam_model_registry["vit_h"](checkpoint=model_path)
		predictor = SamPredictor(sam)
		return predictor
	except Exception:
		return None


def bbox_to_polygon_mask(image_shape, bbox, expand_ratio=0.0):
	"""Fallback: create a rectangular mask from bbox with optional expansion."""
	h, w = image_shape[:2]
	x1, y1, x2, y2 = bbox
	bw, bh = x2 - x1, y2 - y1
	x1e = max(0, int(x1 - expand_ratio * bw))
	y1e = max(0, int(y1 - expand_ratio * bh))
	x2e = min(w - 1, int(x2 + expand_ratio * bw))
	y2e = min(h - 1, int(y2 + expand_ratio * bh))
	mask = np.zeros((h, w), dtype=np.uint8)
	mask[y1e:y2e, x1e:x2e] = 255
	return mask


def overlay_mask(image_rgb, mask, color_bgr, alpha=0.45):
	color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
	colored = np.zeros_like(image_rgb)
	colored[..., 0] = color_rgb[0]
	colored[..., 1] = color_rgb[1]
	colored[..., 2] = color_rgb[2]
	mask_bool = mask > 0
	blended = image_rgb.copy()
	blended[mask_bool] = (alpha * colored[mask_bool] + (1 - alpha) * image_rgb[mask_bool]).astype(np.uint8)
	return blended


def run(yolo_model_path: str, image_path: str, output_dir: str, sam_checkpoint: str = None, conf: float = 0.1):
	# Prepare IO
	os.makedirs(output_dir, exist_ok=True)
	image = cv2.imread(image_path)
	if image is None:
		raise FileNotFoundError(f"Image not found: {image_path}")
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Load YOLO
	model = YOLO(yolo_model_path)
	res = model(image, conf=conf, verbose=False)[0]

	# Load SAM if provided and available
	predictor = load_sam(sam_checkpoint) if sam_checkpoint else None
	if predictor is not None:
		predictor.set_image(image_rgb)

	# Prepare figure
	fig, ax = plt.subplots(1, 1, figsize=(14, 10))
	ax.imshow(image_rgb)
	ax.set_axis_off()

	masks_info = []

	if res.boxes is None or len(res.boxes) == 0:
		print("No detections found.")
	else:
		for idx, det in enumerate(res.boxes):
			bbox = det.xyxy[0].cpu().numpy().astype(int)
			cls_id = int(det.cls[0])
			conf_v = float(det.conf[0])
			cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
			color = COLORS.get(cls_name, (255, 255, 255))

			# Generate mask
			mask = None
			if predictor is not None:
				# Use the bbox as a prompt to SAM2
				x1, y1, x2, y2 = bbox
				box = np.array([x1, y1, x2, y2])
				try:
					masks, scores, _ = predictor.predict(box=box, multimask_output=True)
					# Choose best mask by score
					best_idx = int(np.argmax(scores))
					mask = (masks[best_idx] * 255).astype(np.uint8)
				except Exception:
					mask = None

			# Fallback mask if SAM not available or failed
			if mask is None:
				mask = bbox_to_polygon_mask(image.shape, bbox)

			# Overlay mask
			image_rgb = overlay_mask(image_rgb, mask, color)

			# Draw bbox + label
			x1, y1, x2, y2 = bbox
			rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=(color[2]/255, color[1]/255, color[0]/255), facecolor='none')
			ax.add_patch(rect)
			ax.text(x1, max(0, y1 - 6), f"{cls_name} {conf_v:.2f}", fontsize=9,
					bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.8))

			# Save per-object mask
			mask_name = f"{Path(image_path).stem}_obj{idx:02d}_{cls_name}.png"
			cv2.imwrite(str(Path(output_dir) / mask_name), mask)
			masks_info.append({
				"id": idx,
				"class": cls_name,
				"confidence": conf_v,
				"bbox": bbox.tolist(),
				"mask_path": str(Path(output_dir) / mask_name)
			})

	# Update image in axes to latest overlay
	ax.imshow(image_rgb)
	out_path = Path(output_dir) / f"{Path(image_path).stem}_seg_overlay.jpg"
	plt.tight_layout()
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	plt.show()

	# Save a JSON with masks info
	with open(Path(output_dir) / f"{Path(image_path).stem}_masks.json", "w") as f:
		json.dump(masks_info, f, indent=2)

	print(f"Saved overlay: {out_path}")
	return out_path


def main():
	parser = argparse.ArgumentParser(description="Segmentation visualizer with YOLO + SAM2/fallback")
	parser.add_argument("--yolo-model", required=True)
	parser.add_argument("--image-path", required=True)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--sam-checkpoint", default=None, help="Optional SAM2 checkpoint path")
	parser.add_argument("--conf", type=float, default=0.1)
	args = parser.parse_args()

	run(args.yolo_model, args.image_path, args.output_dir, args.sam_checkpoint, conf=args.conf)


if __name__ == "__main__":
	main()
