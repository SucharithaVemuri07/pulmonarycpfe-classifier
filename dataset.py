import os, random, cv2, tifffile
import pandas as pd
import numpy as np
import tifffile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from collections import Counter
from tranforms import full_tf, roi_tf_fibrosis, roi_tf_patch


def custom_collate(batch):
    """Handles variable-sized ROI batches."""
    rois_list, full_imgs, labels = [], [], []
    for item in batch:
        if len(item) == 3:  # Check if the batch item has 2 or 3 elements
            rois, full_img, label = item
            labels.append(label)
        else:
            rois, full_img = item

        rois_list.append(rois)  # Append to list instead of stacking directly
        full_imgs.append(full_img)

    max_rois = max(r.shape[0] for r in rois_list)
    padded_rois = []
    for rois in rois_list:
        num_rois = rois.shape[0]
        padding_shape = (max_rois - num_rois, *rois.shape[1:])
        padding = torch.zeros(padding_shape, dtype=rois.dtype, device=rois.device)
        padded_rois.append(torch.cat([rois, padding]))
    rois_batch = torch.stack(padded_rois)
    full_batch = torch.stack(full_imgs)
    labels_batch = torch.tensor(labels)

    return rois_batch, full_batch, labels_batch


def extract_rois(pil_img: Image.Image, roi_size: int, stride: int):
    W, H = pil_img.size
    rois = []
    for top in range(0, H - roi_size + 1, stride):
        for left in range(0, W - roi_size + 1, stride):
            rois.append(pil_img.crop((left, top, left+roi_size, top+roi_size)))
    if (H - roi_size) % stride:
        for left in range(0, W - roi_size + 1, stride):
            rois.append(pil_img.crop((left, H-roi_size, left+roi_size, H)))
    if (W - roi_size) % stride:
        for top in range(0, H - roi_size + 1, stride):
            rois.append(pil_img.crop((W-roi_size, top, W, top+roi_size)))
    if (H - roi_size) % stride and (W - roi_size) % stride:
        rois.append(pil_img.crop((W-roi_size, H-roi_size, W, H)))
    return rois

class FibrosisCTDataset(Dataset):
    def __init__(self, root, roi_size=128, num_rois=8, stride=64):
        self.paths, self.labels = [], []
        for sub, lab in [('NORMAL',0),('FIBROSIS',1)]:
            for fn in os.listdir(os.path.join(root, sub)):
                if fn.lower().endswith(('.png','.jpg','.jpeg')):
                    self.paths.append(os.path.join(root, sub, fn)); self.labels.append(lab)
        self.roi_size, self.num_rois, self.stride = roi_size, num_rois, stride

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        full_img = full_tf(img)
        rois = extract_rois(img, self.roi_size, self.stride)
        chosen = random.sample(rois, self.num_rois)
        rois_t = torch.stack([roi_tf_fibrosis(r) for r in chosen])  # [K,3,224,224]
        return rois_t, full_img, label

class EmphysemaPatchDataset(Dataset):
    def __init__(self, patch_dir, csv_path):
        df = pd.read_csv(csv_path, header=0)
        self.files, self.labels = [], []
        for i,lab in enumerate(df['Patch Label']):
            if lab==4: continue
            self.files.append(f"patch{i+1}.tiff"); self.labels.append(int(lab)+1)
        self.patch_dir = patch_dir

    def apply_clahe(self, img):
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_norm)
        return img_clahe

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        arr = tifffile.imread(os.path.join(self.patch_dir, self.files[idx]))
        arr = cv2.normalize(arr,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        img = Image.fromarray(arr).convert('RGB')
        label = self.labels[idx]
        full_img = full_tf(img)
        roi = roi_tf_patch(img)                 # single‑patch ROI
        return torch.stack([roi]), full_img, label  # [1,3,224,224]

class Unlabeled(Dataset):  #Test Data
    def __init__(self, ds): self.ds=ds
    def __len__(self): return len(self.ds)
    def __getitem__(self,i):
        rois, full, _ = self.ds[i]; return rois, full