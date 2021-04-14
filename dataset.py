# code is adapted from 
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://github.com/mtreviso/linear-chain-crf

import torch
from torch.utils.data import Dataset, DataLoader

from model import Const

class SyllablesDataset(Dataset):
    
    def __init__(self, data, _to_ix):
        self.data = data
        self._to_ix = _to_ix
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]
    
    @staticmethod
    def prepare_sequence(seq, stoi):
        return torch.tensor(
            [stoi.get(w, Const.UNK_ID) for w in seq], dtype=torch.long)
    
    def collate_fn(self, batch): # new
        
        bs = len(batch) # number of line in batch
        lengths = [len(item[0]) for item in batch] # lengths of lines in batch
        max_lengths = max(lengths) # lengths of longest line in batch
        n_feature = len(batch[0])-1 # how many features are in each line
        # features, tags, and mask; initially fill torch with PAD_ID
        x1s = torch.full((bs, max_lengths), Const.PAD_ID, dtype=torch.long)
        x2s = x1x.clone().detach() if n_feature > 1 else None # if no 2nd feature
        x3s = x1x.clone().detach() if n_feature > 2 else None # if no 3rd feature
        xtags = torch.full((bs, max_lengths), Const.PAD_TAG_ID, dtype=torch.long)
        # encoder
        *x_to_ix, tag_to_ix = self._to_ix
        x1_to_ix, x2_to_ix, x3_to_ix = (list(x_to_ix) + [None]*2)[:3] # None if no x2, x3
            
        for j in range(bs): # j is index in batch
            x1s_ = self.prepare_sequence(batch[j][0], x1_to_ix)
            x1s[j, : x1s_.shape[0]] = x1s_
            if n_feature > 1:
                x2s_ = self.prepare_sequence(batch[j][1], x2_to_ix)
                x2s[j, : x2s_.shape[0]] = x2s_
            if n_feature > 2:
                x3s_ = self.prepare_sequence(batch[j][2], x3_to_ix)
                x3s[j, : x3s_.shape[0]] = x3s_
            xtags_ = self.prepare_sequence(batch[j][-1], tag_to_ix)
            xtags[j, : xtags_.shape[0]] = xtags_ 
        mask = (xtags != Const.PAD_TAG_ID).float()
        
        return x1s, x2s, x3s, xtags, mask, lengths