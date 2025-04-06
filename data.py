from utils import get_pair, get_pair_paths
from torch.utils.data import Dataset

class ForestDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_fpath, mask_fpath = self.pairs[idx]
        img, mask = get_pair(img_fpath, mask_fpath)
        return img, mask