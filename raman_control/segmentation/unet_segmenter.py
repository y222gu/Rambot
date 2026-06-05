"""
U-Net based segmentation using a trained segmentation_models_pytorch model.

Replaces SAM with a custom-trained U-Net for blood cell / sample segmentation.
"""

from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import cv2

from raman_control.config.logging_config import get_logger

logger = get_logger('unet_segmenter')

# ImageNet normalization constants (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default checkpoint path (relative to project root)
DEFAULT_CHECKPOINT = Path(__file__).parent / "models" / "unet_trained_20260205.pth"


class UNetSegmenter:
    """
    U-Net segmenter using segmentation_models_pytorch.

    Usage:
        segmenter = UNetSegmenter()
        if segmenter.load_model():
            mask = segmenter.segment(image)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        encoder_name: str = "resnet34",
        target_size: int = 512,
        device: Optional[str] = None,
    ):
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self._encoder_name = encoder_name
        self._target_size = target_size
        self._model = None
        self._loaded = False

        # Device setup (deferred import of torch)
        self._device_name = device  # resolved in load_model

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, progress_cb=None) -> bool:
        """Load the U-Net model from checkpoint.

        Args:
            progress_cb: Optional callable(percent: int, message: str) invoked
                at load milestones so a GUI can show progress. Loading torch
                and the weights is slow (several seconds), so this lets the
                caller prove the app isn't frozen.
        """
        def report(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)

        try:
            report(15, "Importing PyTorch…")
            import torch
            import segmentation_models_pytorch as smp

            if not self._checkpoint_path.exists():
                logger.error(f"Checkpoint not found at {self._checkpoint_path}")
                return False

            device = self._device_name
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._device = device

            logger.info(f"Loading U-Net ({self._encoder_name}) from {self._checkpoint_path}")

            report(35, "Building U-Net…")
            self._model = smp.Unet(
                encoder_name=self._encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )

            report(55, "Loading weights…")
            checkpoint = torch.load(self._checkpoint_path, map_location=self._device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()

            self._loaded = True
            report(70, "Model ready")
            logger.info(f"U-Net model loaded on {self._device}")
            return True

        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Run: pip install segmentation-models-pytorch torch")
            return False
        except Exception as e:
            logger.error(f"Failed to load U-Net model: {e}")
            return False

    def segment(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        postprocess: bool = True,
        use_tta: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Segment the image and return a binary mask.

        Args:
            image: Input image (grayscale or RGB/BGR, any size).
            threshold: Binarization threshold for the probability map.
            postprocess: Apply morphological cleanup (largest component, hole fill).
            use_tta: Use test-time augmentation (averages 4 flip variants).

        Returns:
            Binary mask (H, W) with values 0/1, same size as input, or None on failure.
        """
        if not self._loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return None

        try:
            # Prepare RGB image
            rgb = self._to_rgb(image)

            if use_tta:
                mask, confidence = self._predict_tta(rgb, threshold, postprocess)
            else:
                mask, confidence = self._predict(rgb, threshold, postprocess)

            if mask is None or not mask.any():
                logger.info("No object detected in segmentation")
                return None

            logger.info(f"Segmentation complete: confidence={confidence:.3f}, "
                        f"area={int(mask.sum())} px")
            # Return as 0/1 uint8
            return (mask > 0).astype(np.uint8)

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return None

    # ---- Internal ----

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert input image to RGB."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 3:
            return image  # Assume already RGB (frames from camera thread are RGB)
        return image

    def _preprocess(self, image: np.ndarray):
        """Resize, normalize, convert to tensor."""
        import torch

        img = cv2.resize(image, (self._target_size, self._target_size))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)

    def _predict(
        self, image: np.ndarray, threshold: float, postprocess: bool
    ) -> Tuple[Optional[np.ndarray], float]:
        """Single-pass prediction."""
        import torch

        h, w = image.shape[:2]
        input_tensor = self._preprocess(image)

        with torch.no_grad():
            logits = self._model(input_tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        prob = cv2.resize(prob, (w, h))
        mask = (prob > threshold).astype(np.uint8) * 255
        confidence = float(prob[mask > 0].mean()) if mask.any() else 0.0

        if postprocess:
            mask = self._postprocess_mask(mask)

        return mask, confidence

    def _predict_tta(
        self, image: np.ndarray, threshold: float, postprocess: bool
    ) -> Tuple[Optional[np.ndarray], float]:
        """Prediction with test-time augmentation (4 flip variants)."""
        import torch

        h, w = image.shape[:2]

        augmented = [
            image,
            np.fliplr(image).copy(),
            np.flipud(image).copy(),
            np.flipud(np.fliplr(image)).copy(),
        ]

        prob_sum = np.zeros((self._target_size, self._target_size), dtype=np.float32)

        with torch.no_grad():
            for i, aug_img in enumerate(augmented):
                input_tensor = self._preprocess(aug_img)
                logits = self._model(input_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                # Undo augmentation
                if i == 1:
                    prob = np.fliplr(prob)
                elif i == 2:
                    prob = np.flipud(prob)
                elif i == 3:
                    prob = np.flipud(np.fliplr(prob))

                prob_sum += prob

        prob_avg = prob_sum / len(augmented)
        prob_avg = cv2.resize(prob_avg, (w, h))

        mask = (prob_avg > threshold).astype(np.uint8) * 255
        confidence = float(prob_avg[mask > 0].mean()) if mask.any() else 0.0

        if postprocess:
            mask = self._postprocess_mask(mask)

        return mask, confidence

    @staticmethod
    def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
        """Keep largest component, fill holes, morphological cleanup."""
        # Largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255

        # Fill holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, -1)

        # Morphological close then open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask
