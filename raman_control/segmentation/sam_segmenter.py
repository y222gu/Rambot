"""
SAM (Segment Anything Model) based segmentation.

Uses the SAM vit_b model for automatic image segmentation.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from pathlib import Path

from raman_control.config.logging_config import get_logger

logger = get_logger('sam_segmenter')

# Model checkpoint path
MODEL_PATH = Path(__file__).parent / "sam_vit_b_01ec64.pth"


class SAMSegmenter:
    """
    Segment Anything Model (SAM) wrapper for image segmentation.

    Features:
    - Automatic mask generation
    - Point-based prompting
    - Box-based prompting
    - Mask filtering by size

    Usage:
        segmenter = SAMSegmenter()
        segmenter.load_model()

        # Automatic segmentation
        masks = segmenter.segment_automatic(image)

        # Point prompt
        masks = segmenter.segment_point(image, point=(100, 200))

        # Get centroids for sampling
        centroids = segmenter.get_mask_centroids(masks)
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize SAM segmenter.

        Args:
            model_path: Path to SAM checkpoint. Uses default if None.
            device: 'cpu' or 'cuda'
        """
        self._model_path = Path(model_path) if model_path else MODEL_PATH
        self._device = device
        self._model = None
        self._predictor = None
        self._mask_generator = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> bool:
        """Load the SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

            if not self._model_path.exists():
                logger.error(f"SAM model not found at {self._model_path}")
                return False

            logger.info(f"Loading SAM model from {self._model_path}")

            # Load model
            self._model = sam_model_registry["vit_b"](checkpoint=str(self._model_path))
            self._model.to(self._device)

            # Create predictor and mask generator
            self._predictor = SamPredictor(self._model)
            self._mask_generator = SamAutomaticMaskGenerator(
                self._model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=100,
            )

            self._loaded = True
            logger.info("SAM model loaded successfully")
            return True

        except ImportError:
            logger.error("segment_anything package not installed. Run: pip install segment-anything")
            return False
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            return False

    def segment_automatic(
        self,
        image: np.ndarray,
        min_area: int = 100,
        max_area: int = 100000
    ) -> List[Dict[str, Any]]:
        """
        Automatically segment all objects in the image.

        Args:
            image: RGB image (H, W, 3)
            min_area: Minimum mask area to keep
            max_area: Maximum mask area to keep

        Returns:
            List of mask dictionaries with 'segmentation', 'area', 'bbox', etc.
        """
        if not self._loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        try:
            masks = self._mask_generator.generate(image)

            # Filter by area
            filtered_masks = [
                m for m in masks
                if min_area <= m['area'] <= max_area
            ]

            logger.info(f"SAM found {len(masks)} masks, {len(filtered_masks)} after filtering")
            return filtered_masks

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return []

    def segment_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        label: int = 1
    ) -> np.ndarray:
        """
        Segment object at a specific point.

        Args:
            image: RGB image (H, W, 3)
            point: (x, y) coordinates
            label: 1 for foreground, 0 for background

        Returns:
            Binary mask (H, W)
        """
        if not self._loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        try:
            self._predictor.set_image(image)

            masks, scores, logits = self._predictor.predict(
                point_coords=np.array([[point[0], point[1]]]),
                point_labels=np.array([label]),
                multimask_output=True
            )

            # Return best mask
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8)

        except Exception as e:
            logger.error(f"Point segmentation failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def segment_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Segment object within bounding box.

        Args:
            image: RGB image (H, W, 3)
            box: (x1, y1, x2, y2) bounding box

        Returns:
            Binary mask (H, W)
        """
        if not self._loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        try:
            self._predictor.set_image(image)

            masks, scores, logits = self._predictor.predict(
                box=np.array(box),
                multimask_output=True
            )

            # Return best mask
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8)

        except Exception as e:
            logger.error(f"Box segmentation failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)

    @staticmethod
    def get_mask_centroids(masks: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """
        Get centroids of all masks.

        Args:
            masks: List of mask dictionaries from segment_automatic()

        Returns:
            List of (x, y) centroid coordinates
        """
        centroids = []
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0:
                cx = int(np.mean(x_indices))
                cy = int(np.mean(y_indices))
                centroids.append((cx, cy))
        return centroids

    @staticmethod
    def get_mask_bboxes(masks: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes of all masks.

        Returns:
            List of (x, y, width, height) bounding boxes
        """
        return [tuple(m['bbox']) for m in masks]

    def set_generator_params(
        self,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100
    ):
        """Update mask generator parameters."""
        if self._loaded:
            from segment_anything import SamAutomaticMaskGenerator
            self._mask_generator = SamAutomaticMaskGenerator(
                self._model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
            logger.info("SAM generator parameters updated")
