#!/usr/bin/env python3
"""Utility NMS routine for YOLO-style outputs on Raspberry Pi/Hailo."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class Detection:
    """Represents a single detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int

    def to_xywh(self) -> tuple[float, float, float, float, float, int]:
        """Return (cx, cy, w, h, score, class_id)."""
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        cx = self.x1 + w / 2
        cy = self.y1 + h / 2
        return (cx, cy, w, h, self.score, self.class_id)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] boxes to [x1, y1, x2, y2]."""
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return converted


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and multiple boxes."""
    # Intersection coordinates
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - inter + 1e-16
    return inter / union


def non_max_suppression(
    predictions: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    center_format: bool = False,
) -> List[Detection]:
    """Apply class-wise NMS on YOLO-style predictions.

    Args:
        predictions: Array with columns [x1, y1, x2, y2, conf, class]
            or [cx, cy, w, h, conf, class] if center_format is True.
        conf_threshold: Filter detections below this confidence.
        iou_threshold: IoU threshold for suppression.
        max_detections: Maximum number of detections to return.
        center_format: Set True if boxes are in [cx, cy, w, h].

    Returns:
        List of retained detections sorted by score.
    """
    if predictions.size == 0:
        return []

    if predictions.shape[1] != 6:
        raise ValueError("Expected predictions with 6 columns (box, score, class)")

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5].astype(np.int32)

    if center_format:
        boxes = xywh_to_xyxy(boxes)

    keep = scores >= conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    if boxes.size == 0:
        return []

    results: List[Detection] = []
    for cls in np.unique(classes):
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        while cls_boxes.size and len(results) < max_detections:
            box = cls_boxes[0]
            score = cls_scores[0]
            results.append(Detection(box[0], box[1], box[2], box[3], float(score), int(cls)))

            if cls_boxes.shape[0] == 1:
                break

            ious = box_iou(box, cls_boxes[1:])
            keep_indices = np.where(ious <= iou_threshold)[0] + 1
            cls_boxes = cls_boxes[keep_indices]
            cls_scores = cls_scores[keep_indices]

    results.sort(key=lambda det: det.score, reverse=True)
    return results[:max_detections]


def load_predictions(path: str) -> np.ndarray:
    """Load detections stored as a .npy or .txt file for quick testing."""
    if path.endswith((".npy", ".npz")):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "pred" in arr:
                return arr["pred"]
            raise KeyError("NPZ archive must contain an array named 'pred'")
        return arr
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


def format_output(detections: Sequence[Detection]) -> Iterable[str]:
    """Create human-readable strings for CLI output."""
    for det in detections:
        yield (
            f"class={det.class_id:02d} score={det.score:.3f} "
            f"box=({det.x1:.1f},{det.y1:.1f},{det.x2:.1f},{det.y2:.1f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO-style NMS utility for Raspberry Pi/Hailo")
    parser.add_argument("predictions", help="Path to raw predictions (npy/npz/txt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections")
    parser.add_argument(
        "--center-format",
        action="store_true",
        help="Set if boxes are encoded as [cx, cy, w, h]",
    )
    args = parser.parse_args()

    preds = load_predictions(args.predictions)
    detections = non_max_suppression(
        preds,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_detections=args.max_det,
        center_format=args.center_format,
    )

    for line in format_output(detections):
        print(line)


if __name__ == "__main__":
    main()
