#!/usr/bin/env python3
"""
yolo_detector.py — YOLOv8 vehicle detection for traffic density estimation.

Uses YOLOv8n (nano) pretrained on COCO — no custom training required.
COCO vehicle classes used:
  0  : person     weight = 0.5
  2  : car        weight = 1.0
  5  : bus        weight = 3.0
  7  : truck      weight = 2.0

Weighted traffic density formula:
  density = (1.0 × n_cars) + (2.0 × n_trucks) + (3.0 × n_buses) + (0.5 × n_people)
Normalised to [1, 100] to match the RL agent's state space.

Usage:
  detector = YOLODetector()
  result   = detector.detect(image_path="frame.png")
  density  = result["density"]   # int in [1, 100]
  image    = result["annotated"] # PIL Image with bounding boxes drawn
"""

import pathlib
import logging
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs → weights for traffic density
VEHICLE_CLASSES = {
    0: ("person",     0.5),
    2: ("car",        1.0),
    5: ("bus",        3.0),
    7: ("truck",      2.0),
}

# Normalisation cap: density = 100 when raw_density >= this value
MAX_RAW_DENSITY = 50.0

# YOLOv8 model type (nano for speed — no GPU required)
MODEL_NAME = "yolov8n.pt"
CONFIDENCE  = 0.40


class YOLODetector:
    """
    Wraps YOLOv8 inference for traffic density estimation from images.

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
          counts   : {class_name: count}   per vehicle type
          raw      : float                 weighted sum before normalisation
          density  : int [1, 100]          normalised state for RL agent
          annotated: np.ndarray            image with bounding boxes drawn
          num_detections: int
        """
        self._load_model()

        results = self._model(image, conf=self.confidence, verbose=False)[0]

        counts = {name: 0 for _, (name, _) in VEHICLE_CLASSES.items()}
        raw_density = 0.0

        boxes = results.boxes
        if boxes is not None:
            for cls_id, conf in zip(boxes.cls.cpu().numpy(),
                                    boxes.conf.cpu().numpy()):
                cls_id = int(cls_id)
                if cls_id in VEHICLE_CLASSES:
                    name, weight = VEHICLE_CLASSES[cls_id]
                    counts[name] += 1
                    raw_density  += weight

        # Normalise to [1, 100]
        density = int((raw_density / MAX_RAW_DENSITY) * 99) + 1
        density = max(1, min(100, density))

        # Annotated image (BGR ndarray with bboxes)
        annotated = results.plot()

        return {
            "counts":        counts,
            "raw":           round(raw_density, 2),
            "density":       density,
            "annotated":     annotated,
            "num_detections": int(sum(counts.values())),
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
