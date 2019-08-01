import torch
import torch.utils.data as torchdata

from collections import defaultdict
from torchtext import data

class CustomDataset(torchdata.Dataset):
    def __init__(self, path=None):
        with open(path, 'r', encoding='utf-8') as f:
            pass

        
    def __getitem__(self, index):

        return 0

    
    def __len__(self):

        return 0

    
    def custom_collate_fn(self, data):
        
        return 0