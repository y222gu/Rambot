"""
Segmentation panel widget for image segmentation and sampling pattern creation.

Features:
- SAM (Segment Anything) segmentation
- Traditional watershed segmentation
- Segmentation parameter controls
- Sampling pattern generation
- Position list management
"""

from typing import Optional, List
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFrame, QListWidget,
    QListWidgetItem, QSplitter, QProgressBar,
    QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap

from raman_control.gui.styles import COLORS
from raman_control.config.logging_config import get_logger

logger = get_logger('segmentation_panel')


class SegmentationWorker(QThread):
    """Worker thread for running segmentation."""

    progress = pyqtSignal(int, str)  # percent, message
    finished = pyqtSignal(list)  # list of masks
    error = pyqtSignal(str)

    def __init__(self, method: str, image: np.ndarray, params: dict):
        super().__init__()
        self._method = method
        self._image = image
        self._params = params

    def run(self):
        try:
            if self._method == 'sam':
                self._run_sam()
            else:
                self._run_watershed()
        except Exception as e:
            self.error.emit(str(e))

    def _run_sam(self):
        self.progress.emit(10, "Loading SAM model...")

        from raman_control.segmentation import SAMSegmenter
        segmenter = SAMSegmenter()

        if not segmenter.load_model():
            self.error.emit("Failed to load SAM model")
            return

        self.progress.emit(50, "Running segmentation...")

        masks = segmenter.segment_automatic(
            self._image,
            min_area=self._params.get('min_area', 100),
            max_area=self._params.get('max_area', 100000)
        )

        self.progress.emit(100, "Done")
        self.finished.emit(masks)

    def _run_watershed(self):
        self.progress.emit(10, "Running watershed...")

        from raman_control.segmentation import WatershedSegmenter
        segmenter = WatershedSegmenter()

        segmenter.set_parameters(
            blur_size=self._params.get('blur_size', 5),
            min_area=self._params.get('min_area', 100),
            max_area=self._params.get('max_area', 100000),
            distance_threshold=self._params.get('distance_threshold', 0.5)
        )

        self.progress.emit(50, "Processing...")
        masks = segmenter.segment(self._image)
        mask_dicts = segmenter.get_mask_dicts()

        self.progress.emit(100, "Done")
        self.finished.emit(mask_dicts)


class SegmentationPanelWidget(QWidget):
    """
    Panel for image segmentation and sampling pattern creation.

    Signals:
        positions_generated: Emitted when positions are ready (list of (x, y, z))
        segmentation_complete: Emitted when segmentation finishes
    """

    positions_generated = pyqtSignal(list)
    segmentation_complete = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._current_image: Optional[np.ndarray] = None
        self._current_masks: List[dict] = []
        self._worker: Optional[SegmentationWorker] = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # === Image Source ===
        source_group = QGroupBox("Image Source")
        source_layout = QVBoxLayout(source_group)

        btn_layout = QHBoxLayout()

        self._capture_btn = QPushButton("📷 Capture from Camera")
        self._capture_btn.clicked.connect(self._on_capture)
        btn_layout.addWidget(self._capture_btn)

        self._load_btn = QPushButton("📁 Load Image")
        self._load_btn.clicked.connect(self._on_load_image)
        btn_layout.addWidget(self._load_btn)

        source_layout.addLayout(btn_layout)

        self._image_label = QLabel("No image loaded")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumHeight(150)
        self._image_label.setStyleSheet(f'''
            background-color: {COLORS['bg_light']};
            border: 1px solid {COLORS['border_dark']};
            border-radius: 4px;
            color: {COLORS['text_secondary']};
        ''')
        source_layout.addWidget(self._image_label)

        main_layout.addWidget(source_group)

        # === Segmentation Method ===
        method_group = QGroupBox("Segmentation Method")
        method_layout = QVBoxLayout(method_group)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))

        self._method_combo = QComboBox()
        self._method_combo.addItems(["SAM (AI-based)", "Watershed (Traditional)"])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo, 1)

        method_layout.addLayout(method_row)

        # Parameters
        params_layout = QHBoxLayout()

        params_layout.addWidget(QLabel("Min Area:"))
        self._min_area_spin = QSpinBox()
        self._min_area_spin.setRange(10, 100000)
        self._min_area_spin.setValue(100)
        self._min_area_spin.setSuffix(" px²")
        params_layout.addWidget(self._min_area_spin)

        params_layout.addWidget(QLabel("Max Area:"))
        self._max_area_spin = QSpinBox()
        self._max_area_spin.setRange(100, 1000000)
        self._max_area_spin.setValue(100000)
        self._max_area_spin.setSuffix(" px²")
        params_layout.addWidget(self._max_area_spin)

        method_layout.addLayout(params_layout)

        # Watershed-specific params
        self._watershed_params = QWidget()
        ws_layout = QHBoxLayout(self._watershed_params)
        ws_layout.setContentsMargins(0, 0, 0, 0)

        ws_layout.addWidget(QLabel("Blur:"))
        self._blur_spin = QSpinBox()
        self._blur_spin.setRange(1, 21)
        self._blur_spin.setValue(5)
        self._blur_spin.setSingleStep(2)
        ws_layout.addWidget(self._blur_spin)

        ws_layout.addWidget(QLabel("Threshold:"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.1, 1.0)
        self._threshold_spin.setValue(0.5)
        self._threshold_spin.setSingleStep(0.1)
        ws_layout.addWidget(self._threshold_spin)

        ws_layout.addStretch()
        self._watershed_params.hide()
        method_layout.addWidget(self._watershed_params)

        # Run button
        self._run_btn = QPushButton("▶ Run Segmentation")
        self._run_btn.setProperty('primary', True)
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._on_run_segmentation)
        self._run_btn.setEnabled(False)
        method_layout.addWidget(self._run_btn)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        method_layout.addWidget(self._progress_bar)

        main_layout.addWidget(method_group)

        # === Results ===
        results_group = QGroupBox("Detected Objects")
        results_layout = QVBoxLayout(results_group)

        self._results_label = QLabel("No segmentation results")
        self._results_label.setStyleSheet(f'color: {COLORS["text_secondary"]};')
        results_layout.addWidget(self._results_label)

        # Position list
        self._position_list = QListWidget()
        self._position_list.setMaximumHeight(150)
        self._position_list.setStyleSheet(f'''
            QListWidget {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border_dark']};
                border-radius: 4px;
            }}
        ''')
        results_layout.addWidget(self._position_list)

        # Generate positions button
        btn_row = QHBoxLayout()

        self._generate_btn = QPushButton("Generate Sample Positions")
        self._generate_btn.clicked.connect(self._on_generate_positions)
        self._generate_btn.setEnabled(False)
        btn_row.addWidget(self._generate_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._clear_btn)

        results_layout.addLayout(btn_row)

        main_layout.addWidget(results_group)

        main_layout.addStretch()

    def _on_method_changed(self, index: int):
        """Show/hide method-specific parameters."""
        self._watershed_params.setVisible(index == 1)

    def _on_capture(self):
        """Capture current camera frame."""
        # This will be connected to camera in main window
        logger.info("Capture requested")

    def _on_load_image(self):
        """Load image from file."""
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )

        if path:
            import cv2
            image = cv2.imread(path)
            if image is not None:
                self.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def set_image(self, image: np.ndarray):
        """Set the image to segment."""
        self._current_image = image
        self._display_image(image)
        self._run_btn.setEnabled(True)
        logger.info(f"Image set: {image.shape}")

    def _display_image(self, image: np.ndarray):
        """Display image in preview."""
        if image.ndim == 2:
            h, w = image.shape
            qimage = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, c = image.shape
            qimage = QImage(image.data, w, h, w * c, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)

    def _on_run_segmentation(self):
        """Run segmentation on current image."""
        if self._current_image is None:
            return

        method = 'sam' if self._method_combo.currentIndex() == 0 else 'watershed'

        params = {
            'min_area': self._min_area_spin.value(),
            'max_area': self._max_area_spin.value(),
            'blur_size': self._blur_spin.value(),
            'distance_threshold': self._threshold_spin.value(),
        }

        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)

        self._worker = SegmentationWorker(method, self._current_image, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_segmentation_finished)
        self._worker.error.connect(self._on_segmentation_error)
        self._worker.start()

    def _on_progress(self, percent: int, message: str):
        """Update progress bar."""
        self._progress_bar.setValue(percent)
        self._results_label.setText(message)

    def _on_segmentation_finished(self, masks: list):
        """Handle segmentation completion."""
        self._current_masks = masks
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)

        count = len(masks)
        self._results_label.setText(f"Found {count} objects")
        self._generate_btn.setEnabled(count > 0)

        # Update list
        self._position_list.clear()
        for i, mask in enumerate(masks):
            area = mask.get('area', 0)
            item = QListWidgetItem(f"Object {i+1}: {area} px²")
            self._position_list.addItem(item)

        self.segmentation_complete.emit()
        logger.info(f"Segmentation complete: {count} objects")

    def _on_segmentation_error(self, error: str):
        """Handle segmentation error."""
        self._progress_bar.setVisible(False)
        self._run_btn.setEnabled(True)
        self._results_label.setText(f"Error: {error}")
        logger.error(f"Segmentation error: {error}")

        QMessageBox.warning(self, "Segmentation Error", error)

    def _on_generate_positions(self):
        """Generate sample positions from masks."""
        if not self._current_masks:
            return

        from raman_control.segmentation import PatternGenerator

        generator = PatternGenerator()
        positions = generator.from_masks(self._current_masks)

        position_list = generator.get_position_list()
        self.positions_generated.emit(position_list)

        logger.info(f"Generated {len(position_list)} sample positions")

    def _on_clear(self):
        """Clear results."""
        self._current_masks = []
        self._position_list.clear()
        self._results_label.setText("No segmentation results")
        self._generate_btn.setEnabled(False)
