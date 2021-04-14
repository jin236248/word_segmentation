import torch, json
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from util import word_tokenisation as wt # for evaluation testset
from util.util import *
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
    
    def collate_fn(self, batch):
        
        this_bs = len(batch)
        max_sent_size = max([len(item[0]) for item in batch])

        x1s = torch.full((this_bs, max_sent_size), Const.PAD_ID, dtype=torch.long).cuda()
        x2s, x3s = x1s, x1s # same
        xtags = torch.full((this_bs, max_sent_size), Const.PAD_TAG_ID, dtype=torch.long).cuda()
        lengths = []
        n_data = len(batch[0])-1 # how many x
            
        if n_data == 1:

            x1_to_ix, tag_to_ix = self._to_ix
            
            for j in range(this_bs): # j is index in batch
                
                x1s_ = prepare_sequence(batch[j][0], x1_to_ix)
                x1s[j, : x1s_.shape[0]] = x1s_

                xtags_ = prepare_sequence(batch[j][-1], tag_to_ix)
                xtags[j, : xtags_.shape[0]] = xtags_ 
                
                lengths.append(len(x1s_))
                
            return x1s, xtags, lengths
            
        if n_data == 2:

            x1_to_ix, x2_to_ix, tag_to_ix = self._to_ix
            
            for j in range(this_bs): # j is index in batch
                
                x1s_ = prepare_sequence(batch[j][0], x1_to_ix)
                x1s[j, : x1s_.shape[0]] = x1s_
                
                x2s_ = prepare_sequence(batch[j][1], x2_to_ix)
                x2s[j, : x2s_.shape[0]] = x2s_
                
                xtags_ = prepare_sequence(batch[j][-1], tag_to_ix)
                xtags[j, : xtags_.shape[0]] = xtags_ 
                
                lengths.append(len(x1s_))
                
            return x1s, x2s, xtags, lengths
            
        if n_data == 3:

            x1_to_ix, x2_to_ix, x3_to_ix, tag_to_ix = self._to_ix
            
            for j in range(this_bs): # j is index in batch
                
                x1s_ = prepare_sequence(batch[j][0], x1_to_ix)
                x1s[j, : x1s_.shape[0]] = x1s_
                
                x2s_ = prepare_sequence(batch[j][1], x2_to_ix)
                x2s[j, : x2s_.shape[0]] = x2s_
                
                x3s_ = prepare_sequence(batch[j][2], x3_to_ix)
                x3s[j, : x3s_.shape[0]] = x3s_
                
                xtags_ = prepare_sequence(batch[j][-1], tag_to_ix)
                xtags[j, : xtags_.shape[0]] = xtags_ 
                
                lengths.append(len(x1s_))
                
            return x1s, x2s, x3s, xtags, lengths

def prepare_sequence(seq, stoi):
    return torch.tensor([stoi.get(w, Const.UNK_ID) for w in seq], dtype=torch.long).cuda()

def ids_to_tags(seq, itos):
    return [itos[x] for x in seq]

def train(model, _to_ix, n_epoch, bs, learn_rate, train_data, validate_data, 
          best_score, name_to_save='', modeltype='sy', shuffle=True, drop=True):
    
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, weight_decay=1e-4)
    dataset = SyllablesDataset(train_data, _to_ix)
    onetenth = max(1, len(dataset) // (bs * 10))
    
    for epoch in range(n_epoch):
        begin = datetime.now()
        print(f'epoch: {epoch}, progress: ', end='')
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, collate_fn=dataset.collate_fn)
        
        n_data = len(train_data[0])-1
        
        for i, data in enumerate(dataloader): # index to select batch from all_training_data
            
            x2s, x3s = None, None
            if n_data == 1: x1s, xtags, lengths = data
            if n_data == 2: x1s, x2s, xtags, lengths = data
            if n_data == 3: x1s, x2s, x3s, xtags, lengths = data
                
            mask = (xtags != Const.PAD_TAG_ID).float()
            model.zero_grad()
            _, _, loss = model(x1s, x2s, x3s, xtags, mask=mask, drop=drop)
            loss.backward()
            optimizer.step()
            show_progress(i, onetenth)
            
        report_time(begin)

        # validate
        pred_tags, correct_tags, _, _ = get_tags(model, _to_ix, dataset, bs)
        score = accuracy_score(correct_tags, pred_tags) * 100
        print(f'score: {score:.2f}')
        
        if name_to_save != '' and score > best_score:
            best_score = score
            torch.save(model.state_dict(), 'model/' + name_to_save + '.pth')
                        
    return best_score
                    
def get_tags(model, _to_ix, dataset, bs):
    
    dataset = SyllablesDataset(dataset, _to_ix)
    with torch.no_grad():
        pred_tags_notflat, pred_tags, correct_tags  = [],[],[]
        dataloader = DataLoader(dataset, batch_size=bs, collate_fn=dataset.collate_fn)
        n_data = len(dataset[0])-1

        for i, data in enumerate(dataloader):
            
            if n_data == 1: 
                x1s, xtags, lengths = data
                mask = (x1s != Const.PAD_ID).float()
                _, seqs, _ = model(x1s, None, None, xtags, mask=mask, drop=False)
                
            if n_data == 2: 
                x1s, x2s, xtags, lengths = data
                mask = (x1s != Const.PAD_ID).float()
                _, seqs, _ = model(x1s, x2s, None, xtags, mask=mask, drop=False)
            
            if n_data == 3: 
                x1s, x2s, x3s, xtags, lengths = data
                mask = (x1s != Const.PAD_ID).float()
                _, seqs, _ = model(x1s, x2s, x3s, xtags, mask=mask, drop=False)
                
            pred_tags_notflat += [seq[:length] for seq, length in zip(seqs, lengths)] 
            pred_tags = [tag for seqs in pred_tags_notflat for tag in seqs] 
            correct_tags += [x for y in xtags.tolist() for x in y if x != Const.PAD_TAG_ID] # targets without pad
            
        return pred_tags, correct_tags, lengths, pred_tags_notflat
    

def build_pred_text(_to_ix, test_data, pred_tags, modeltype='sy'):

    tag_to_ix = _to_ix[-1]
    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
    
    # if model is bigram, we choose only the first char in bigram to build pred_text
    if modeltype=='bi': 
        test_data = [([bi[0] for bi in bis], tags) for bis, tags in test_data]
    elif modeltype=='chctbi': 
        test_data = [(chs, tags) for chs, cts, bis, tags in test_data]
    elif modeltype=='bictsy': 
        test_data = [([bi[0] for bi in bis], tags) for bis, cts, sys, tags in test_data]

    newsent = ''
    for i, tags in enumerate(pred_tags): # each line
        newsent += test_data[i][0][0] # first syllable
        
        for j in range(1, len(tags)): # each next syllable
            
            if ix_to_tag[tags[j]][0] == 'B': 
                newsent += '|' + test_data[i][0][j]
            else: 
                newsent += test_data[i][0][j]
        
        newsent += '|' + '\n' # at the end of each line
    newsent = newsent[:-2] + '\n' # correction at the end
    
    return newsent

def evaluate(model, _to_ix, test_data, bs, modeltype='sy'):
    
    with open('dataset/answer_text.txt') as file: 
        answer_text = file.read() # answer text

    _, _, _, pred_tags = get_tags(model, _to_ix, test_data, bs=bs) # run the model to get prediction
    pred_text = build_pred_text(_to_ix, test_data, pred_tags, modeltype=modeltype) # build pred_text
    
    result = wt._compute_stats(answer_text, pred_text)
    char_level = str({k:v for k,v in result['char_level'].items() if k in ['precision', 'recall', 'f1']})
    word_level = str(result['word_level'])
    print('char level:' + char_level + '\nword level:' + word_level)
    
    return result, pred_text, answer_text