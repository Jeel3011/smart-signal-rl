#!/usr/bin/env python3
"""
yolo_detector.py — YOLOv8 vehicle detection for traffic density estimation.

Uses YOLOv8n (nano) pretrained on COCO — no custom training required.

IMPORTANT — Aerial / top-down image support
============================================
YOLOv8n is trained on *street-level* COCO images.  When the camera is
overhead (drone, CCTV pole, SUMO screenshot), vehicles look very different
from their side-view appearance.  COCO-trained models frequently
misclassify aerial cars as "cell phone", "suitcase", "book", etc. because
the rectangular top-down silhouette matches those classes better.

This detector uses a **hybrid strategy**:
  1.  Standard COCO class filter — catches vehicles detected correctly.
  2.  Bounding-box area + aspect-ratio heuristic — catches ANY detection
      whose bounding box is the right *size and shape* for a vehicle in
      the image, regardless of the predicted COCO class label.

The two sets are unioned (no double-counting) to produce the final
vehicle count used for density estimation.

COCO vehicle classes used (when detected correctly):
  0  : person     weight = 0.5
  1  : bicycle    weight = 0.8
  2  : car        weight = 1.0
  3  : motorcycle weight = 0.8
  5  : bus        weight = 3.0
  7  : truck      weight = 2.0

Weighted traffic density formula:
  density = Σ (weight × count)   normalised to [1, 100]

Usage:
  detector = YOLODetector()
  result   = detector.detect(image_path="frame.png")
  density  = result["density"]   # int in [1, 100]
  image    = result["annotated"] # numpy array with bounding boxes drawn
"""

import pathlib
import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── COCO class IDs → (name, density_weight) ──────────────────────────────
VEHICLE_CLASSES = {
    0: ("person",      0.5),   # pedestrians
    1: ("bicycle",     0.8),   # cyclists
    2: ("car",         1.0),   # cars (also catches auto-rickshaws)
    3: ("motorcycle",  0.8),   # bikes + pillion riders counted as one box
    5: ("bus",         3.0),   # buses
    7: ("truck",       2.0),   # trucks
}

# COCO classes that aerial vehicles are commonly misclassified as.
# We DON'T count these by class label — but we DON'T reject their boxes
# from the area-heuristic either.  Listing them here for documentation.
AERIAL_MISCLASS_IDS = {
    39,  # bottle
    41,  # cup
    56,  # chair
    63,  # laptop
    64,  # mouse
    66,  # keyboard
    67,  # cell phone
    73,  # book
    58,  # potted plant
    28,  # suitcase
    26,  # handbag
    24,  # backpack
}

# ── Auto-rickshaw aspect ratio heuristic ─────────────────────────────────
AUTO_RICKSHAW_AR_MIN = 0.55
AUTO_RICKSHAW_AR_MAX = 1.15
AUTO_RICKSHAW_WEIGHT = 1.5   # between car(1.0) and truck(2.0)

# ── Aerial vehicle bounding-box heuristic thresholds ─────────────────────
# Expressed as fractions of the image area.
# A "vehicle" in an aerial image typically occupies between 0.2% and 15%
# of the total image area, with an aspect ratio (w/h) between 0.3 and 3.5.
AERIAL_VEHICLE_AREA_MIN_FRAC = 0.002   # 0.2% of image area
AERIAL_VEHICLE_AREA_MAX_FRAC = 0.15    # 15%  of image area
AERIAL_VEHICLE_AR_MIN        = 0.25    # very tall / narrow still ok
AERIAL_VEHICLE_AR_MAX        = 4.0     # very wide / short still ok
AERIAL_VEHICLE_DEFAULT_WEIGHT = 1.0    # count as ≈ 1 car

# ── Normalisation cap ────────────────────────────────────────────────────
MAX_RAW_DENSITY = 50.0

# ── Model defaults ───────────────────────────────────────────────────────
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.20   # lowered to 0.20 — aerial detections often have lower
                     # confidence since the model hasn't seen this viewpoint


class YOLODetector:
    """
    Wraps YOLOv8 inference for traffic density estimation from images.

    Supports both street-level AND aerial/top-down viewpoints via a
    hybrid class-filter + area-heuristic approach.

    The detector is stateless — call .detect() with any image path or array.
    Model weights are downloaded automatically on first use (~6MB for nano).
    """

    def __init__(self, model_name: str = MODEL_NAME,
                 confidence: float = CONFIDENCE):
        self.model_name = model_name
        self.confidence = confidence
        self._model     = None   # lazy-load on first inference

    def _load_model(self):
        """Lazy-load YOLOv8 model (downloads weights on first call)."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_name)
                logger.info(f"[YOLO] Model loaded: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "ultralytics not installed. Run: pip install ultralytics"
                )

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, image: Union[str, pathlib.Path, np.ndarray]) -> dict:
        """
        Run YOLOv8 on an image and return vehicle counts + density.

        Parameters
        ----------
        image : str, Path, or np.ndarray
            Image to process.

        Returns
        -------
        dict with keys:
          counts        : {class_name: count}   per vehicle type
          raw           : float                  weighted sum before normalisation
          density       : int [1, 100]           normalised state for RL agent
          annotated     : np.ndarray             image with bounding boxes drawn
          num_detections: int
          detection_mode: str                    "coco" | "aerial_heuristic" | "hybrid"
        """
        self._load_model()

        results = self._model(image, conf=self.confidence, verbose=False)[0]

        counts = {name: 0 for _, (name, _) in VEHICLE_CLASSES.items()}
        counts["auto-rickshaw"] = 0   # extra class via heuristic
        raw_density = 0.0

        # Track which box indices were already counted via COCO class filter
        coco_counted_indices = set()

        # Get image dimensions for area-fraction calculations
        img_h, img_w = results.orig_shape[:2]
        img_area = img_h * img_w

        boxes = results.boxes
        all_boxes_data = []   # store (cls_id, conf, xyxy) for second pass

        # ── PASS 1: Standard COCO vehicle-class filter ───────────────
        if boxes is not None and len(boxes) > 0:
            cls_arr  = boxes.cls.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            xyxy_arr = boxes.xyxy.cpu().numpy()

            for idx, (cls_id, conf, xyxy) in enumerate(
                    zip(cls_arr, conf_arr, xyxy_arr)):

                cls_id = int(cls_id)
                all_boxes_data.append((idx, cls_id, conf, xyxy))

                if cls_id not in VEHICLE_CLASSES:
                    continue

                name, weight = VEHICLE_CLASSES[cls_id]

                # ── Auto-rickshaw heuristic ───────────────────────────
                if cls_id in (2, 7):   # car or truck
                    x1, y1, x2, y2 = xyxy
                    w, h = (x2 - x1), (y2 - y1)
                    ar   = w / max(h, 1)
                    if AUTO_RICKSHAW_AR_MIN <= ar <= AUTO_RICKSHAW_AR_MAX:
                        counts["auto-rickshaw"] += 1
                        raw_density += AUTO_RICKSHAW_WEIGHT
                        coco_counted_indices.add(idx)
                        continue   # don't double-count as car

                counts[name] += 1
                raw_density  += weight
                coco_counted_indices.add(idx)

        coco_vehicle_count = sum(counts.values())

        # ── PASS 2: Aerial bounding-box area heuristic ───────────────
        # For any detection NOT already counted by COCO class, check if
        # the bounding box is "vehicle-shaped" based on area fraction
        # and aspect ratio.  This catches cars misclassified as
        # "cell phone", "book", "potted plant", etc.
        aerial_extra_count = 0
        for (idx, cls_id, conf, xyxy) in all_boxes_data:
            if idx in coco_counted_indices:
                continue   # already counted

            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            box_area = w * h
            area_frac = box_area / max(img_area, 1)
            ar = w / max(h, 1)

            # Check if this box is vehicle-sized
            if (AERIAL_VEHICLE_AREA_MIN_FRAC <= area_frac <= AERIAL_VEHICLE_AREA_MAX_FRAC
                    and AERIAL_VEHICLE_AR_MIN <= ar <= AERIAL_VEHICLE_AR_MAX):
                aerial_extra_count += 1
                raw_density += AERIAL_VEHICLE_DEFAULT_WEIGHT
                coco_counted_indices.add(idx)
                logger.debug(
                    f"[YOLO] Aerial heuristic: box {idx} (class={cls_id}, "
                    f"conf={conf:.2f}) counted as vehicle "
                    f"(area_frac={area_frac:.4f}, ar={ar:.2f})"
                )

        # Add aerial extras to car count (best approximation)
        if aerial_extra_count > 0:
            counts["car"] += aerial_extra_count
            logger.info(
                f"[YOLO] Aerial heuristic added {aerial_extra_count} "
                f"extra vehicle(s) from misclassified detections"
            )

        # Determine detection mode for diagnostics
        if coco_vehicle_count > 0 and aerial_extra_count > 0:
            detection_mode = "hybrid"
        elif aerial_extra_count > 0:
            detection_mode = "aerial_heuristic"
        else:
            detection_mode = "coco"

        total_vehicles = sum(counts.values())

        # Normalise to [1, 100]
        density = int((raw_density / MAX_RAW_DENSITY) * 99) + 1
        density = max(1, min(100, density))

        # Annotated image (BGR ndarray with bboxes)
        annotated = results.plot()

        return {
            "counts":          counts,
            "raw":             round(raw_density, 2),
            "density":         density,
            "annotated":       annotated,
            "num_detections":  total_vehicles,
            "detection_mode":  detection_mode,
        }

    def density_from_counts(self, counts: dict) -> int:
        """
        Compute density from a pre-supplied counts dict.
        Useful when counts come from SUMO TraCI (for training mode).
        """
        raw  = sum(counts.get(name, 0) * weight
                   for _, (name, weight) in VEHICLE_CLASSES.items())
        density = int((raw / MAX_RAW_DENSITY) * 99) + 1
        return max(1, min(100, density))

    def detect_batch(self, images: list) -> list[dict]:
        """Run detection on a list of images."""
        return [self.detect(img) for img in images]

    def save_annotated(self, result: dict, path: Union[str, pathlib.Path]):
        """Save annotated image to disk."""
        import cv2
        cv2.imwrite(str(path), result["annotated"])
        logger.info(f"[YOLO] Saved annotated image → {path}")


# ---------------------------------------------------------------------------
# Demo / standalone test
# ---------------------------------------------------------------------------

def demo_on_sample_image(image_path: Optional[str] = None):
    """
    Run detector on a sample image and print results.
    Downloads a sample traffic image if none provided.
    """
    import urllib.request
    import tempfile, cv2

    if image_path is None:
        # Download a CC0 traffic intersection image for demo
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
        # Use a local traffic image if it exists
        sample = pathlib.Path("sumo_env/screenshots")
        imgs = list(sample.glob("*.png")) if sample.exists() else []
        if imgs:
            image_path = str(imgs[0])
        else:
            print("No sample image found. Run SUMO demo first to capture screenshots.")
            return

    detector = YOLODetector()
    result   = detector.detect(image_path)

    print(f"\n{'='*50}")
    print(f"  YOLO Detection Results")
    print(f"  Image: {image_path}")
    print(f"  Detection mode: {result.get('detection_mode', 'unknown')}")
    print(f"  {'─'*40}")
    for cls, cnt in result["counts"].items():
        print(f"    {cls:<10}: {cnt}")
    print(f"  {'─'*40}")
    print(f"  Raw density  : {result['raw']:.2f}")
    print(f"  RL State     : {result['density']} / 100")
    print(f"  Total dets.  : {result['num_detections']}")
    print(f"{'='*50}\n")

    # Save annotated
    out = pathlib.Path("results/yolo_demo.png")
    out.parent.mkdir(exist_ok=True)
    import cv2
    cv2.imwrite(str(out), result["annotated"])
    print(f"  Annotated image saved → {out}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, help="Path to image for detection")
    args = p.parse_args()
    demo_on_sample_image(args.image)
