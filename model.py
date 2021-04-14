import torch
import torch.nn as nn
import torch.nn.functional as F # for emb drop out
from torch.autograd import Variable # for emb drop out

# code is adapted from 
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://github.com/mtreviso/linear-chain-crf

class Const:
    
    UNK_ID, UNK_TOKEN = 0, "<unk>"
    PAD_ID, PAD_TOKEN = 1, "<pad>"
    BOS_ID, BOS_TOKEN = 2, "<bos>"
    EOS_ID, EOS_TOKEN = 3, "<eos>"
    PAD_TAG_ID, PAD_TAG_TOKEN = 0, "<pad>"
    BOS_TAG_ID, BOS_TAG_TOKEN = 1, "<bos>"
    EOS_TAG_ID, EOS_TAG_TOKEN = 2, "<eos>"
    
class BiLSTM_CRF(nn.Module):
    
    def __init__(self, x1vocab_size, x2vocab_size=0, x3vocab_size=0, label_size=0, x1emb_dim=512, x2emb_dim=0, x3emb_dim=0, hidden_dim=1024):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.x1emb = nn.Embedding(x1vocab_size, x1emb_dim)
        if x2vocab_size>0: self.x2emb = nn.Embedding(x2vocab_size, x2emb_dim)
        if x3vocab_size>0: self.x3emb = nn.Embedding(x3vocab_size, x3emb_dim)
        
        total_emb_dim = x1emb_dim
        if x2vocab_size>0: total_emb_dim += x2emb_dim
        if x3vocab_size>0: total_emb_dim += x3emb_dim
        self.lstm = nn.LSTM(total_emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        
        self.hidden2tag = nn.Linear(hidden_dim, label_size)
        self.hidden = None
        self.dropout = nn.Dropout(p=0.2)
        self.nb_labels = label_size
        self.BOS_TAG_ID = Const.BOS_TAG_ID
        self.EOS_TAG_ID = Const.EOS_TAG_ID
        self.PAD_TAG_ID = Const.PAD_TAG_ID
        self.batch_first = True
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

        if self.PAD_TAG_ID is not None:
            self.transitions.data[self.PAD_TAG_ID, :] = -10000.0 # no transitions from padding
            self.transitions.data[:, self.PAD_TAG_ID] = -10000.0 # no transitions to padding
            self.transitions.data[self.PAD_TAG_ID, self.EOS_TAG_ID] = 0.0 # except if the end of sentence is reached or we are already in a pad position
            self.transitions.data[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).cuda(), torch.randn(2, batch_size, self.hidden_dim // 2).cuda())
    
    def embed_dropout(self, embed, words, dropout=0.1):
        
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight

        padding_idx = embed.padding_idx
        if padding_idx is None: padding_idx = -1
        X = F.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

        return X

    def compute_scores(self, emissions, tags, mask):
        
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size).cuda()

        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        t_scores = self.transitions[self.BOS_TAG_ID, first_tags]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores

        for i in range(1, seq_length):
            is_valid = mask[:, i]
            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]
            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid
            scores += e_scores + t_scores
        scores += self.transitions[last_tags, self.EOS_TAG_ID]

        return scores
    
    def find_best_path(self, sample_id, best_tag, backpointers):

        best_path = [best_tag]
        for backpointers_t in reversed(backpointers):
            best_tag = backpointers_t[best_tag][sample_id].item()
            best_path.insert(0, best_tag)

        return best_path
    
    def forward(self, x1s, x2s, x3s, tags, mask=None, drop=True):
        
        self.hidden = self.init_hidden(x1s.shape[0])
        
        if drop == True:
            x1 = self.embed_dropout(self.x1emb, x1s)
            if x2s != None: x2 = self.embed_dropout(self.x2emb, x2s)
            if x3s != None: x3 = self.embed_dropout(self.x3emb, x3s)
        else:
            x1 = self.x1emb(x1s)
            if x2s != None: x2 = self.x2emb(x2s)
            if x3s != None: x3 = self.x3emb(x3s)
                
        x = x1
        if x2s != None: x = torch.cat((x1, x2), dim=2)
        if x3s != None: x = torch.cat((x1, x2, x3), dim=2)
        
        x, self.hidden = self.lstm(x, self.hidden)
        if drop == True: x = self.dropout(x) 
        emissions = self.hidden2tag(x)
        if mask is None: mask = torch.ones(emissions.shape[:2], dtype=torch.float)
            
        batch_size, seq_length, nb_labels = emissions.shape
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]
        backpointers = []

        for i in range(1, seq_length):
            e_scores = emissions[:, i].unsqueeze(1)
            t_scores = self.transitions.unsqueeze(0)
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            max_scores, max_score_tags = torch.max(scores, dim=1)
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas
            backpointers.append(max_score_tags.t())

        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        
        for i in range(batch_size):
            sample_length = emission_lengths[i].item()
            sample_final_tag = max_final_tags[i].item()
            sample_backpointers = backpointers[: sample_length - 1]
            sample_path = self.find_best_path(i, sample_final_tag, sample_backpointers)
            best_sequences.append(sample_path)
            
        partition = torch.logsumexp(end_scores, dim=1)
        gold_scores = self.compute_scores(emissions, tags, mask=mask)        
        loss = torch.sum(partition - gold_scores)
    
        return max_final_scores, best_sequences, loss
        