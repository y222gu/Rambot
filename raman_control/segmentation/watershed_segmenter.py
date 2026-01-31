"""
Traditional watershed-based image segmentation.

Provides a simpler, faster alternative to SAM for well-defined objects.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

from raman_control.config.logging_config import get_logger

logger = get_logger('watershed_segmenter')


class WatershedSegmenter:
    """
    Traditional watershed segmentation pipeline.

    Features:
    - Preprocessing (blur, threshold)
    - Distance transform
    - Watershed segmentation
    - Configurable parameters

    Usage:
        segmenter = WatershedSegmenter()
        masks = segmenter.segment(image)
        centroids = segmenter.get_centroids()
    """

    def __init__(self):
        # Parameters
        self._blur_size = 5
        self._threshold_method = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        self._min_area = 100
        self._max_area = 100000
        self._distance_threshold = 0.5

        # Results
        self._labels: Optional[np.ndarray] = None
        self._masks: List[np.ndarray] = []
        self._centroids: List[Tuple[int, int]] = []
        self._areas: List[int] = []

    def set_parameters(
        self,
        blur_size: int = 5,
        min_area: int = 100,
        max_area: int = 100000,
        distance_threshold: float = 0.5
    ):
        """
        Set segmentation parameters.

        Args:
            blur_size: Gaussian blur kernel size (odd number)
            min_area: Minimum object area to keep
            max_area: Maximum object area to keep
            distance_threshold: Fraction of max distance for marker detection
        """
        self._blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self._min_area = min_area
        self._max_area = max_area
        self._distance_threshold = distance_threshold

    def segment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Segment objects in the image using watershed.

        Args:
            image: Input image (grayscale or RGB)

        Returns:
            List of binary masks, one per detected object
        """
        self._masks = []
        self._centroids = []
        self._areas = []

        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Preprocessing: blur and threshold
            blurred = cv2.GaussianBlur(gray, (self._blur_size, self._blur_size), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, self._threshold_method)

            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Sure background area (dilated)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Distance transform for sure foreground
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(
                dist_transform,
                self._distance_threshold * dist_transform.max(),
                255, 0
            )
            sure_fg = np.uint8(sure_fg)

            # Unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labeling
            _, markers = cv2.connectedComponents(sure_fg)

            # Add 1 to all labels so background is 1, not 0
            markers = markers + 1

            # Mark unknown region with 0
            markers[unknown == 255] = 0

            # Apply watershed
            if len(image.shape) == 2:
                color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                color_image = image.copy()

            self._labels = cv2.watershed(color_image, markers)

            # Extract individual masks
            unique_labels = np.unique(self._labels)
            for label in unique_labels:
                if label <= 1:  # Skip background (1) and boundary (-1)
                    continue

                mask = (self._labels == label).astype(np.uint8) * 255
                area = np.sum(mask > 0)

                # Filter by area
                if self._min_area <= area <= self._max_area:
                    self._masks.append(mask)
                    self._areas.append(area)

                    # Calculate centroid
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) > 0:
                        cx = int(np.mean(x_indices))
                        cy = int(np.mean(y_indices))
                        self._centroids.append((cx, cy))

            logger.info(f"Watershed found {len(self._masks)} objects")
            return self._masks

        except Exception as e:
            logger.error(f"Watershed segmentation failed: {e}")
            return []

    def get_centroids(self) -> List[Tuple[int, int]]:
        """Get centroids of detected objects."""
        return self._centroids.copy()

    def get_areas(self) -> List[int]:
        """Get areas of detected objects."""
        return self._areas.copy()

    def get_labels(self) -> Optional[np.ndarray]:
        """Get label image from watershed."""
        return self._labels

    def get_mask_dicts(self) -> List[Dict[str, Any]]:
        """
        Get masks in SAM-compatible format.

        Returns:
            List of dictionaries with 'segmentation', 'area', 'centroid'
        """
        results = []
        for i, mask in enumerate(self._masks):
            results.append({
                'segmentation': mask > 0,
                'area': self._areas[i] if i < len(self._areas) else 0,
                'centroid': self._centroids[i] if i < len(self._centroids) else (0, 0),
            })
        return results

    @staticmethod
    def draw_overlay(
        image: np.ndarray,
        masks: List[np.ndarray],
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw segmentation overlay on image.

        Args:
            image: Original image
            masks: List of binary masks
            alpha: Overlay transparency
            color: Overlay color (B, G, R)

        Returns:
            Image with overlay
        """
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()

        overlay = output.copy()

        for mask in masks:
            overlay[mask > 0] = color

        return cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    @staticmethod
    def draw_centroids(
        image: np.ndarray,
        centroids: List[Tuple[int, int]],
        radius: int = 5,
        color: Tuple[int, int, int] = (0, 0, 255)
    ) -> np.ndarray:
        """Draw centroid markers on image."""
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()

        for cx, cy in centroids:
            cv2.circle(output, (cx, cy), radius, color, -1)
            cv2.circle(output, (cx, cy), radius + 2, (255, 255, 255), 1)

        return output
