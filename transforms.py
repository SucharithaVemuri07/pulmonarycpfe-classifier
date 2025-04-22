from PIL import Image
import numpy as np
from torchvision import transforms
import cv2

class CropImageTransform:
    def __init__(self, top=0.15, bottom=0.08, right=0.02):
        self.top, self.bottom, self.right = top, bottom, right
    def __call__(self, img: Image.Image):
        C,H,W = 3,*img.size[::-1]
        if H < 200 or W < 200:
            return img
        t,b,r = int(H*self.top), int(H*self.bottom), int(W*self.right)
        return img.crop((0+t*0, t, W-r, H-b))  # keep full left side

class ContrastNormalize:
    def __call__(self, image):
        img_np = np.array(image).astype(np.float32)
        img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        return Image.fromarray(img_uint8)

class MorphologyErosion:
    def __call__(self, image):
        img_np = np.array(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(img_np, kernel, iterations=1)
        return Image.fromarray(eroded)

full_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

roi_tf_fibrosis = transforms.Compose([
    CropImageTransform(crop_right_ratio=0.02, crop_bottom_ratio=0.07, crop_top_ratio=0.08),
    transforms.Resize((224,224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    ContrastNormalize(),
    transforms.ToTensor(),
])

roi_tf_patch = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])