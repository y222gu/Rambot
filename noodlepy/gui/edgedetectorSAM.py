import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import os
from PIL import Image
import io
import torch
import cv2 as cv
from threading import Thread

class EdgeDetectorSAM():
    def __init__(self, image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self.marker_size = image.shape[1]/2
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        model_path = os.path.join(os.getcwd(), "sam_vit_h_4b8939.pth")
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
        if torch.cuda.is_available():
            self.sam.to('cuda')

    def auto_mask_generate(self):
        generator = SamAutomaticMaskGenerator(model=self.sam)
        masks = generator.generate(self.image)

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width*px, self.height*px))
        ax.imshow(self.image)
        self.show_anns(masks, ax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')  # Save the figure to the buffer
        buf.seek(0)  # Rewind the buffer
        pil_image = Image.open(buf)

        plt.close(fig)
        return pil_image


    def point_prompt_mask_generate(self, label=1):
        # put the point in the center of the image
        prompt_x, prompt_y = self.center_x, self.center_y

        input_point = np.array([[prompt_x, prompt_y]])
        input_label = np.array([label])
        generator = SamPredictor(self.sam)
        generator.set_image(self.image)
        masks, scores, logits = generator.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        # second mask is the usually best mask
        self.best_mask = masks[1]
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width*px, self.height*px))
        ax.imshow(self.image)
        self.show_mask(self.best_mask, ax)
        # self.show_points(input_point, input_label, ax, '*', marker_size=self.marker_size)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')  # Save the figure to the buffer
        buf.seek(0)  # Rewind the buffer
        pil_image = Image.open(buf)
        # save the figure
        save_path = os.path.join(os.getcwd(), "masked_image.png")
        # ensure subsequent pil_image.save uses high DPI (e.g., 600x600)
        orig_save = pil_image.save
        def _save_with_dpi(path, *args, **kwargs):
            kwargs.setdefault('dpi', (600, 600))
            return orig_save(path, *args, **kwargs)
        pil_image.save = _save_with_dpi
        base_name, ext = os.path.splitext(save_path)
        index = 1
        while os.path.exists(save_path):
            save_path = f"{base_name}_{index}{ext}"
            index += 1
        pil_image.save(save_path)
        plt.close(fig)
        return pil_image

    def box_prompt_mask_generate(self, x_low=950, y_low=180, x_high=1400, y_high=620):
        input_box = np.array([x_low, y_low, x_high, y_high])
        generator = SamPredictor(self.sam)
        generator.set_image(self.image)
        masks, scores, logits = generator.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width*px, self.height*px))
        ax.imshow(self.image)
        self.show_mask(masks[0], ax)
        self.show_box(input_box, ax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')  # Save the figure to the buffer
        buf.seek(0)  # Rewind the buffer
        pil_image = Image.open(buf)

        plt.close(fig)
        return pil_image
        

    def _compute_buffered_mask(self, mask, buffer):
        """Return a boolean mask of the region where sampling points are allowed.

        `buffer` is in pixels and measures how far the outermost points sit from
        the detected sample edge:
          buffer > 0  -> points kept at least `buffer` px inside the edge
          buffer == 0 -> points kept inside the mask (original behavior)
          buffer < 0  -> points may spill up to `|buffer|` px outside the edge
        """
        mask = (mask > 0).astype(np.uint8)
        if buffer == 0:
            return mask.astype(bool)
        # Signed distance from the edge: positive inside the mask, negative outside.
        dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        signed_dist = dist_inside - dist_outside
        return signed_dist >= buffer

    @staticmethod
    def _resample_closed_contour(pts, n):
        """Resample a closed contour into `n` points spaced equally by arc length.

        `pts` is an (N, 2) array of (x, y) contour vertices; the loop is treated as
        closed (last vertex connects back to the first). Returns an (n, 2) array.
        """
        if n <= 0 or len(pts) == 0:
            return np.empty((0, 2))
        if len(pts) == 1:
            return np.repeat(pts, n, axis=0)
        # Close the loop and measure cumulative arc length along the polygon
        closed = np.vstack([pts, pts[:1]])
        seg_lengths = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cumulative[-1]
        if total_length == 0:
            return np.repeat(pts[:1], n, axis=0)
        # Equally spaced targets around the perimeter, then interpolate x and y
        targets = np.linspace(0, total_length, n, endpoint=False)
        rx = np.interp(targets, cumulative, closed[:, 0])
        ry = np.interp(targets, cumulative, closed[:, 1])
        return np.stack([rx, ry], axis=1)

    def generate_sampling_points(self, shape, **kwargs):
        coordinates_order = None

        h, w = self.best_mask.shape[-2:]
        mask = self.best_mask.reshape(h, w)
        mask = mask > 0.5
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # Buffer distance (px) of the outermost points from the detected edge.
        # Fall back to the legacy 'offset_from_the_edge' keyword if 'buffer' is absent.
        buffer = kwargs.get('buffer', kwargs.get('offset_from_the_edge', 0))
        allowed = self._compute_buffered_mask(mask, buffer)

        sampling_y, sampling_x = np.where(allowed)

        if shape == 'grid':
            row_number, col_number = kwargs['row_number'], kwargs['col_number']

            # Build grid over the mask bounding box
            sampling_x = np.linspace(sampling_x.min(), sampling_x.max(), int(col_number))
            sampling_y = np.linspace(sampling_y.min(), sampling_y.max(), int(row_number))
            sampling_x, sampling_y = np.meshgrid(sampling_x, sampling_y)
            sampling_x, sampling_y = sampling_x.flatten(), sampling_y.flatten()

            # --- keep only points inside the buffered (allowed) region ---
            ix = np.clip(np.round(sampling_x).astype(int), 0, self.width - 1)
            iy = np.clip(np.round(sampling_y).astype(int), 0, self.height - 1)
            inside = allowed[iy, ix]

            sampling_x = sampling_x[inside]
            sampling_y = sampling_y[inside]

            # compute grid row/col indices for each candidate point, then keep only those inside
            col_indices = np.tile(np.arange(col_number), row_number)
            row_indices = np.repeat(np.arange(row_number), col_number)
            col_indices = col_indices[inside]
            row_indices = row_indices[inside]

            # convert to 1-based indexing for human-friendly coordinates order
            coordinates_order = [{'row': int(r) + 1, 'col': int(c) + 1} for r, c in zip(row_indices, col_indices)]

            # save the filtered coordinates order and the corresponding sampling points to a text file for debugging
            with open("filtered_coordinates_order_and_sampling_points.txt", "w") as f:
                f.write("Coordinates Order (Row, Col) and Corresponding Sampling Points (X, Y):\n")
                for r, c, x, y in zip(row_indices + 1, col_indices + 1, sampling_x, sampling_y):
                    f.write(f"Row: {int(r)}, Col: {int(c)} -> X: {float(x)}, Y: {float(y)}\n")

        elif shape == 'random':
            num_points = kwargs['num_points']
            indices = np.random.choice(len(sampling_x), num_points, replace=False)
            sampling_x, sampling_y = sampling_x[indices], sampling_y[indices]

        elif shape == 'line':
            num_points = kwargs['num_points']

            # ensure the allowed region has positive pixels
            if not allowed.any():
                sampling_x, sampling_y = np.array([]), np.array([])
            else:
                h_img, w_img = allowed.shape
                # centroid of allowed pixels
                ys, xs = np.where(allowed)
                cx = int(np.round(xs.mean()))
                cy = int(np.round(ys.mean()))

                # search outward from centroid for a row with allowed pixels
                found_row = None
                for offset in range(h_img):
                    for r in (cy + offset, cy - offset) if offset else (cy,):
                        if 0 <= r < h_img:
                            cols = np.where(allowed[r])[0]
                            if cols.size:
                                # pick the largest contiguous segment in this row
                                diffs = np.diff(cols)
                                splits = np.where(diffs > 1)[0]
                                segments = []
                                start = 0
                                for s in splits:
                                    segments.append(cols[start:s+1])
                                    start = s + 1
                                segments.append(cols[start:])
                                largest = max(segments, key=len)
                                xmin, xmax = float(largest[0]), float(largest[-1])
                                sampling_x = np.linspace(xmin, xmax, num_points)
                                sampling_y = np.full_like(sampling_x, float(r))
                                found_row = r
                                break
                    if found_row is not None:
                        break

                # fallback if no row found (use full mask horizontal span)
                if found_row is None:
                    xmin, xmax = float(xs.min()), float(xs.max())
                    sampling_x = np.linspace(xmin, xmax, num_points)
                    sampling_y = np.full_like(sampling_x, float(self.center_y))

        elif shape == 'rings':
            # True concentric contours: points sit on iso-distance contours at fixed
            # depths from the detected edge and are sampled at equal arc length, so the
            # spacing stays even and consistent even for irregular / concave samples.
            num_points_per_contour = kwargs.get('num_rays', kwargs.get('num_points', 50))  # points per contour
            num_contours = kwargs['num_rings']  # number of concentric contours
            interval = kwargs['interval']       # depth spacing between contours (px)
            # `buffer` is the depth of the OUTERMOST contour from the edge (negative spills out)

            if not mask.any():
                sampling_x, sampling_y = np.array([]), np.array([])
                ring_numbers = np.array([])
                line_numbers = np.array([])
            else:
                # Signed distance field: positive inside the mask, negative outside.
                m = (mask > 0).astype(np.uint8)
                dist_inside = cv2.distanceTransform(m, cv2.DIST_L2, 5)
                dist_outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
                signed_dist = dist_inside - dist_outside

                sampling_x, sampling_y = [], []
                ring_numbers, line_numbers = [], []

                for contour_idx in range(int(num_contours)):
                    # Depth of this contour from the edge (outermost first, going inward)
                    level = buffer + contour_idx * interval
                    # The boundary of the "at least `level` deep" region is the iso-contour
                    region = (signed_dist >= level).astype(np.uint8)
                    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # Keep contours with a meaningful perimeter (a shape can split into
                    # several pieces at a given depth, e.g. concave / annular samples)
                    contours = [c for c in contours if len(c) >= 2 and cv2.arcLength(c, True) > 1]
                    if not contours:
                        continue
                    perimeters = np.array([cv2.arcLength(c, True) for c in contours])
                    total_perimeter = perimeters.sum()

                    line_counter = 0
                    for c, perimeter in zip(contours, perimeters):
                        # distribute the points across pieces proportional to their perimeter
                        n_pts = max(1, int(round(num_points_per_contour * perimeter / total_perimeter)))
                        resampled = self._resample_closed_contour(c.reshape(-1, 2).astype(float), n_pts)
                        for rx, ry in resampled:
                            sampling_x.append(float(rx))
                            sampling_y.append(float(ry))
                            ring_numbers.append(contour_idx + 1)
                            line_counter += 1
                            line_numbers.append(line_counter)

                sampling_x = np.array(sampling_x)
                sampling_y = np.array(sampling_y)
                ring_numbers = np.array(ring_numbers)
                line_numbers = np.array(line_numbers)

                coordinates_order = [{'ring': int(r), 'line': int(l)} for r, l in zip(ring_numbers, line_numbers)]

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width*px, self.height*px))
        ax.imshow(self.image)
        self.show_mask(self.best_mask, ax)
        self.show_points(np.stack([sampling_x, sampling_y], axis=1), np.ones(len(sampling_x)), ax, '.', marker_size=self.marker_size)
        
        # Add text labels for ring and radial numbers if shape is 'rings'
        if shape == 'rings':
            for i, (x, y) in enumerate(zip(sampling_x, sampling_y)):
                label = f"R{ring_numbers[i]}:r{line_numbers[i]}"
                ax.text(x, y + 15, label, fontsize=8, ha='center', color='white', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pil_image = Image.open(buf)
        plt.close(fig)
        return pil_image, sampling_x, sampling_y, coordinates_order

    @staticmethod
    def find_edge_of_eroded_mask(mask, erosion_size=10, erosion_shape=cv.MORPH_RECT):
        """
        Finds the edge coordinates of an eroded mask.
        
        Parameters:
            mask (np.ndarray): A binary matrix (2D numpy array) where the area of interest is labeled with 1.
            erosion_size (int): Size of the erosion kernel. Default is 3.
            erosion_shape (int): Shape of the erosion kernel. Options are cv.MORPH_RECT, cv.MORPH_CROSS, and cv.MORPH_ELLIPSE.
                                Default is cv.MORPH_RECT.
        
        Returns:
            edge_coords (np.ndarray): Coordinates of the edges in the eroded mask.
        """
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
        eroded_mask = cv.erode(mask, element)
        edges = cv.Canny(eroded_mask, 0, 1)
        edge_coords = np.argwhere(edges > 0)
        return edge_coords


    @staticmethod
    def show_anns(anns, ax=None):
        if len(anns) == 0:
            return
        if ax is None:
            ax = plt.gca()
            ax.set_autoscale_on(False)
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.5)))

    @staticmethod
    def show_points(coords, labels, ax, marker, marker_size):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='#f39c12', marker=marker, s=marker_size, linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker=marker, s=marker_size, linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(), "livecamera_captured_frame.png")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    edge_detector = EdgeDetectorSAM(image)
    pil_image = edge_detector.point_prompt_mask_generate()
    pil_image.show()
    save_path = "masked_image.png"
    base_name, ext = os.path.splitext(save_path)
    index = 1
    while os.path.exists(save_path):
        save_path = f"{base_name}_{index}{ext}"
        index += 1
    pil_image.save(save_path)
    print(f"Mask generated successfully! Saved as {save_path}")

    # Example usage with rings (now creates radial lines with points)
    image, x, y = edge_detector.generate_sampling_points(
        shape='rings', 
        num_points=50,           # Number of radial lines (you can also use num_rays)
        num_rings=5,             # Number of points along each radial line
        interval=20,             # Pixels between each ring
        offset_from_the_edge=10  # Start 10 pixels from the edge
    )