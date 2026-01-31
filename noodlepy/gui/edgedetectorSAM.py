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
        

    def generate_sampling_points(self, shape, **kwargs):
        coordinates_order = None

        h, w = self.best_mask.shape[-2:]
        mask = self.best_mask.reshape(h, w)
        mask = mask > 0.5
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        sampling_y, sampling_x = np.where(mask == 1)

        if shape == 'grid':
            row_number, col_number = kwargs['row_number'], kwargs['col_number']

            # Build grid over the mask bounding box
            sampling_x = np.linspace(sampling_x.min(), sampling_x.max(), int(col_number))
            sampling_y = np.linspace(sampling_y.min(), sampling_y.max(), int(row_number))
            sampling_x, sampling_y = np.meshgrid(sampling_x, sampling_y)
            sampling_x, sampling_y = sampling_x.flatten(), sampling_y.flatten()

            # --- NEW: keep only points inside the mask ---
            ix = np.clip(np.round(sampling_x).astype(int), 0, self.width - 1)
            iy = np.clip(np.round(sampling_y).astype(int), 0, self.height - 1)
            inside = mask[iy, ix] == 1

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

            # ensure mask has positive pixels
            if not mask.any():
                sampling_x, sampling_y = np.array([]), np.array([])
            else:
                h_img, w_img = mask.shape
                # centroid of mask pixels
                ys, xs = np.where(mask == 1)
                cx = int(np.round(xs.mean()))
                cy = int(np.round(ys.mean()))

                # search outward from centroid for a row with mask pixels
                found_row = None
                for offset in range(h_img):
                    for r in (cy + offset, cy - offset) if offset else (cy,):
                        if 0 <= r < h_img:
                            cols = np.where(mask[r] == 1)[0]
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
            # Radial lines with points distributed from near-center to edge
            num_rays = kwargs.get('num_rays', kwargs.get('num_points', 50))  # Number of radial lines
            num_rings = kwargs['num_rings']  # Points along each ray
            interval = kwargs['interval']  # Pixel interval between rings
            offset_from_the_edge = kwargs['offset_from_the_edge']  # Offset from edge in pixels
            
            # Find mask center and edge distances
            ys, xs = np.where(mask == 1)
            if len(ys) == 0:
                sampling_x, sampling_y = np.array([]), np.array([])
                ring_numbers = np.array([])
                line_numbers = np.array([])
            else:
                center_x = xs.mean()
                center_y = ys.mean()
                
                sampling_x, sampling_y = [], []
                ring_numbers, line_numbers = [], []
                
                # Generate rays at equal angular intervals
                angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
                
                for ray_idx, angle in enumerate(angles):
                    # Direction vector for this ray
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    
                    # Find intersection of ray with mask boundary
                    max_dist = max(self.width, self.height)
                    edge_dist = 0
                    
                    for dist in range(int(max_dist)):
                        x = int(center_x + dist * dx)
                        y = int(center_y + dist * dy)
                        
                        # Check if we've left the mask
                        if x < 0 or x >= self.width or y < 0 or y >= self.height or mask[y, x] == 0:
                            edge_dist = dist - 1
                            break
                    
                    # Calculate positions for points along this ray
                    if edge_dist > 0:
                        # Start from offset_from_the_edge pixels from the edge
                        # and place points at interval pixels apart going inward
                        start_dist = edge_dist - offset_from_the_edge
                        
                        # Minimum distance from center to avoid overlap (e.g., 5-10 pixels)
                        min_dist_from_center = 10
                        
                        # Generate point positions
                        point_distances = []
                        for i in range(num_rings):
                            dist = start_dist - i * interval
                            if dist > min_dist_from_center:
                                point_distances.append(dist)
                        
                        # Add points along this ray
                        for ring_idx, d in enumerate(point_distances):
                            px = center_x + d * dx
                            py = center_y + d * dy
                            # Only add if inside mask
                            if 0 <= int(px) < self.width and 0 <= int(py) < self.height:
                                if mask[int(py), int(px)] > 0:
                                    sampling_x.append(px)
                                    sampling_y.append(py)
                                    ring_numbers.append(ring_idx + 1)
                                    line_numbers.append(ray_idx + 1)
                
                sampling_x = np.array(sampling_x)
                sampling_y = np.array(sampling_y)
                ring_numbers = np.array(ring_numbers)
                line_numbers = np.array(line_numbers)

                # Compute angles and sort
                point_angles = np.arctan2(sampling_y - center_y, sampling_x - center_x)
                sorted_indices = np.argsort(point_angles)

                # Sort all arrays by angle
                sampling_x = sampling_x[sorted_indices]
                sampling_y = sampling_y[sorted_indices]
                ring_numbers = ring_numbers[sorted_indices]
                line_numbers = line_numbers[sorted_indices]

                # Rotate arrays so the sequence starts from a chosen reference point
                # (keep the original radial numbers so all points on the same ray keep the same id)
                min_y_index = np.argmin(sampling_y)
                sampling_x = np.roll(sampling_x, -min_y_index)
                sampling_y = np.roll(sampling_y, -min_y_index)
                ring_numbers = np.roll(ring_numbers, -min_y_index)
                line_numbers = np.roll(line_numbers, -min_y_index)
                
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