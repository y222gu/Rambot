"""
Image segmentation module for Raman microscopy.

Provides:
- SAM (Segment Anything Model) based segmentation
- Traditional watershed segmentation
- Sampling pattern generation from masks
"""

from .sam_segmenter import SAMSegmenter
from .watershed_segmenter import WatershedSegmenter
from .pattern_generator import PatternGenerator

__all__ = ['SAMSegmenter', 'WatershedSegmenter', 'PatternGenerator']
