# code is adapted from 
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://github.com/mtreviso/linear-chain-crf

import torch, json
import torch.optim as optim
from pandas.core.common import flatten
from itertools import chain
from sklearn.metrics import accuracy_score

import pytorch_lightning as pl

from model import *
from dataset import *
from util import get_tags, evaluate

# continue from model.py: train, val, test dataloaders; optimizer; train, validation, test steps
class BiLSTM_CRF_PL(BiLSTM_CRF):

    # load data
    def prepare_data(self):
        with open('dataset/' + self.data_name + '.json') as f: 
            self.train_data, self._to_ix, self.validate_data, self.test_data = json.load(f)
        # common args used in defining dataloader
        self.kwargs = {'batch_size': self.hparams.batch_size, 'num_workers': 20}

    # dataloader    
    def train_dataloader(self):
        dataset = SyllablesDataset(self.train_data, self._to_ix)
        return DataLoader(dataset, collate_fn=dataset.collate_fn, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        dataset = SyllablesDataset(self.train_data[:2000], self._to_ix)
        loader0 = DataLoader(dataset, collate_fn=dataset.collate_fn, **self.kwargs)
        dataset = SyllablesDataset(self.validate_data, self._to_ix)
        loader1 = DataLoader(dataset, collate_fn=dataset.collate_fn, **self.kwargs)
        return [loader0, loader1]
    
    def test_dataloader(self):
        dataset = SyllablesDataset(self.test_data, self._to_ix)
        return DataLoader(dataset, collate_fn=dataset.collate_fn, **self.kwargs)
    
    # optimizer
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.hparams.lr, weight_decay=1e-4)
        return optimizer
    
    # train
    def training_step(self, batch, batch_idx):
        x1s, x2s, x3s, xtags, mask, lengths = batch 
        seqs, loss = self.forward(x1s, x2s, x3s, xtags, mask, drop=True)
        self.log('train_loss', loss)
        return loss
    
    # val
    def validation_step(self, batch, batch_idx, dataset_idx):
        x1s, x2s, x3s, xtags, mask, lengths = batch 
        seqs, loss = self.forward(x1s, x2s, x3s, xtags, mask)
        self.log('val_loss', loss)
        return get_tags(seqs, xtags, lengths)
    
    def validation_epoch_end(self, val_step_outputs):
        for name, outputs in zip(('train', 'val'), val_step_outputs):
            pred_tags, tags = zip(*outputs)
            score = accuracy_score(list(flatten(pred_tags)), list(flatten(tags))) * 100
            self.log(f'{name}_acc', score, prog_bar=True)
    
    # test
    def test_step(self, batch, batch_idx):
        x1s, x2s, x3s, xtags, mask, lengths = batch 
        seqs, loss = self.forward(x1s, x2s, x3s, xtags, mask)
        return get_tags(seqs, xtags, lengths)
    
    def test_epoch_end(self, test_step_outputs):
        pred_tags, tags = zip(*test_step_outputs)
        result = evaluate(
            self._to_ix, self.test_data, list(chain(*pred_tags)), self.m_type)
        self.log('char precision', result['char_level']['precision'])
        self.log('char recall', result['char_level']['recall'])
        self.log('char f1', result['char_level']['f1'])
        self.log('word precision', result['word_level']['precision'])
        self.log('word recall', result['word_level']['recall'])
        self.log('word f1', result['word_level']['f1'])