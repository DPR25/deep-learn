import torch
import numpy as np
import cv2
from utils import tweak_batch, break_into_patches, reconstruct_patch, fill_mask

class SegmentationInference():
    def __init__(self, model, transform, checkpoint_path, device='cuda'):
        self.model = model
        self.model.eval()
        self.transform = transform
        self.checkpoint_path = checkpoint_path
        self.device = device
        if self.checkpoint_path:
            self.load_model()
        self.model.to(device)

    def load_model(self):
        state_dict = torch.load(self.checkpoint_path) 
        self.model.load_state_dict(state_dict)

    def __call__(self, img_path):
        patches, cropped_img, patch_factors = break_into_patches(img_path)
        if patches == None:
            print(f"Image {img_path} had one invalid width and/or height (too small)")
            return None

        # feedforward through the segmentation model
        if self.transform:
            patches = self.transform(patches)

        # form reconstructed segmentation mask from patches
        patches = patches.to(self.device)
        patch_preds, _ = self.model(patches)
        path_pred_mask = torch.argmax(patch_preds, dim=1).cpu().numpy()
        reconstructed_seg_mask = reconstruct_patch(path_pred_mask, patch_factors)

        # downscale to 512 x 512 and send that through the model
        downscaled_img = np.transpose(cropped_img, (1,2,0))
        downscaled_img = cv2.resize(downscaled_img, (512, 512), interpolation=cv2.INTER_AREA)
        downscaled_img = torch.Tensor(downscaled_img).float().permute(2,0,1).unsqueeze(0).to(self.device)
        downscaled_preds, _ = self.model(downscaled_img)
        downscale_pred_mask = torch.argmax(downscaled_preds, dim=1).permute(1,2,0).cpu().numpy()
        upscaled_pred_mask = cv2.resize(downscale_pred_mask, (patch_factors[0] * 512, patch_factors[1] * 512), 
                                        interpolation=cv2.INTER_NEAREST) 

        # final prediction is union of both masks (this should avoid false negatives)
        union_mask = np.logical_or(reconstructed_seg_mask, upscaled_pred_mask)

        # send it through a filter to fill small holes (not interesting for deforestation)
        filled_mask = fill_mask(union_mask).astype(np.int8)

        # also return tweaked cropped img
        tweaked_img = tweak_batch( np.expand_dims(cropped_img, axis=0) )[0]

        return filled_mask, cropped_img