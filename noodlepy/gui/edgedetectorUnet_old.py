import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
import os
import io
from PIL import Image

class EdgeDetectorUnet:
    def __init__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.height, self.width = self.image.shape[:2]
        self.best_mask = None
        self.masked_image = None

    def point_prompt_mask_generate(self, option='largest_area', confidence_threshold=0.01):
        input_tensor = self.transform(self.image).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(input_tensor)

        if len(prediction[0]['masks']) == 0:
            print("No masks detected.")
            self.masked_image = None
            return None
        
        masks = prediction[0]['masks']
        scores = prediction[0]['scores']
        
        if option == 'largest_area':
            valid_masks = [masks[i, 0].cpu().numpy() > 0.5 for i in range(len(scores)) if scores[i] > confidence_threshold]
            if not valid_masks:
                best_mask_idx = torch.argmax(scores).item()
                self.best_mask = masks[best_mask_idx, 0].cpu().numpy() > 0.5
            else:
                self.best_mask = max(valid_masks, key=lambda x: np.sum(x))
        elif option == 'highest_confidence':
            best_mask_idx = torch.argmax(scores).item()
            self.best_mask = masks[best_mask_idx, 0].cpu().numpy() > 0.5
        else:
            raise ValueError("Invalid option. Choose 'largest_area' or 'highest_confidence'.")

        self.masked_image = self.plot_mask()
        return self.masked_image
        
    def plot_mask(self):
        if self.best_mask is None:
            print("No mask available to plot.")
            return None
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        self.show_mask(self.best_mask, plt.gca())
        plt.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pil_image = Image.open(buf)
        pil_image.save(os.path.join(os.getcwd(), "masked_image.png"))
        plt.close()
        return pil_image

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        color = np.concatenate([np.random.rand(3), [0.6]]) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
        mask_image = np.zeros((*mask.shape, 4))
        mask_image[mask] = color
        ax.imshow(mask_image)
