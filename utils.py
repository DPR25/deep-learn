import os

import rasterio
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
import cv2

# training utils
def get_pair_paths(img_fpath, label_fpath):
    label_fnames = os.listdir(label_fpath)
    pairs = []
    for img_fname in os.listdir(img_fpath):
        if img_fname in label_fnames:
            pairs.append((os.path.join(img_fpath, img_fname), os.path.join(label_fpath, img_fname)))
    return pairs

def get_image(image_path):
    with rasterio.open(image_path) as src:
        img = src.read()
    # assumes B04, B03, B02 order (RGB)
    normalized_image = [normalize(band) for band in img[:3,:,:]]
    return np.stack(normalized_image, axis=0)

def get_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read()
    return mask

def get_pair(image_path, label_path):
    return get_image(image_path), get_mask(label_path)

# color tweaking utils (for visualization not training, with exception of normalize())
def brighten(band):
    alpha=0.13
    beta=0
    return np.clip(alpha*band+beta, 0,255)

def gammacorr(band):
    gamma=2
    return np.power(band, 1/gamma)

def normalize(band):
    _min, _max = band.min(), band.max()
    if _min != _max:
        return (band - _min) / (_max - _min)
    return np.zeros(band.shape)

def tweak_band(band):
    band = brighten(band)
    band = gammacorr(band)
    band = normalize(band)
    return band

def tweak_batch(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    tweaked_batch = []
    for img in x:
        tweaked_image = np.stack([tweak_band(b) for b in img], axis=0)
        tweaked_batch.append(tweaked_image)
    return np.stack(tweaked_batch, axis=0)

# inference utils
def open_image(img_path):
    with Image.open(img_path) as img:
        img = np.array(img)
        img = img[:,:,:3]
    return np.transpose(img, (2,0,1))

def break_into_patches(img_path, train_dim=512):
    img = open_image(img_path)

    h, w = img.shape[-2:]
    num_h, num_w = h // train_dim, w // train_dim
    
    # image is too small for processing
    if num_h < 1 or num_w < 1:
        return None, None

    # form patches
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            patch = img[:, i*train_dim : (i+1) * train_dim, j*train_dim : (j+1) * train_dim]

            # need to normalize the patch
            normalized_patch = np.stack([normalize(band) for band in patch], axis=0)
            patches.append(normalized_patch)
    patches = torch.Tensor(np.stack(patches, axis=0)).float()

    # also return cropped image and (num_h, num_w)
    cropped_img =  img[:, 0:num_h * train_dim, 0:num_w * train_dim]

    return patches, cropped_img, (num_h, num_w)

def reconstruct_patch(patch, patch_factors):
    num_h, num_w = patch_factors
    rows = []
    for i in range(num_h):
        row = []
        for j in range(num_w):
            row.append(patch[i * num_w + j])
        rows.append( np.concatenate(row, axis=1) )
            
    return np.concatenate(rows, axis=0)

def fill_mask(seg_mask, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filled_mask = cv2.morphologyEx(seg_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return filled_mask