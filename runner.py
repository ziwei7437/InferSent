# Glue Task Runner Classes.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


import logging
from tqdm import tqdm, trange


# Todo...

class GlueTaskRunner:

    def __init__(self, model, optimizer, tokenizer, label_list, device, rparams):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label_map = {v: i for i, v in enumerate(label_list)}
        self.device = device
        self.rparams = rparams
    
    
