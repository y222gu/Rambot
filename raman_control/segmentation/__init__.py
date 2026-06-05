"""
Image segmentation module for Raman microscopy.

Provides:
- U-Net based segmentation (custom trained)
- SAM (Segment Anything Model) based segmentation
- Traditional watershed segmentation
- Sampling pattern generation from masks
"""

from .unet_segmenter import UNetSegmenter
from .sam_segmenter import SAMSegmenter
from .watershed_segmenter import WatershedSegmenter
from .pattern_generator import PatternGenerator

__all__ = ['UNetSegmenter', 'SAMSegmenter', 'WatershedSegmenter', 'PatternGenerator']
