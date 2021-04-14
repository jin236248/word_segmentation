import re, json, os
from collections import Counter
from ssg import syllable_tokenize as stok
from model import Const
from util.util import *

def get_vocabs(train_data):
    
    # begin with some constant
    x1_to_ix = {Const.UNK_TOKEN: Const.UNK_ID, 
                Const.PAD_TOKEN: Const.PAD_ID, 
                Const.BOS_TOKEN: Const.BOS_ID, 
                Const.EOS_TOKEN: Const.EOS_ID}
    
    x2_to_ix, x3_to_ix = x1_to_ix, x1_to_ix # same
    
    tag_to_ix = {Const.PAD_TAG_TOKEN: Const.PAD_TAG_ID, 
                 Const.BOS_TAG_TOKEN: Const.BOS_TAG_ID, 
                 Const.EOS_TAG_TOKEN: Const.EOS_TAG_ID}

    n_data = len(train_data[0])-1
    
    if n_data == 1:
    
        for x1s, tags in train_data:
        
            for x1, tag in zip(x1s, tags):
                
                if x1 not in x1_to_ix: x1_to_ix[x1] = len(x1_to_ix)
                if tag not in tag_to_ix: tag_to_ix[tag] = len(tag_to_ix) 
                    
        return [x1_to_ix, tag_to_ix]
            
    if n_data == 2:
        
        for x1s, x2s, tags in train_data:
            
            for x1, x2, tag in zip(x1s, x2s, tags):
                  
                if x1 not in x1_to_ix: x1_to_ix[x1] = len(x1_to_ix)
                if x2 not in x2_to_ix: x2_to_ix[x2] = len(x2_to_ix)
                if tag not in tag_to_ix: tag_to_ix[tag] = len(tag_to_ix) 
                    
        return [x1_to_ix, x2_to_ix, tag_to_ix]
            
    if n_data == 3:
        
        for x1s, x2s, x3s, tags in train_data:
            
            for x1, x2, x3, tag in zip(x1s, x2s, x3s, tags):
                  
                if x1 not in x1_to_ix: x1_to_ix[x1] = len(x1_to_ix)
                if x2 not in x2_to_ix: x2_to_ix[x2] = len(x2_to_ix)
                if x3 not in x3_to_ix: x3_to_ix[x3] = len(x3_to_ix)
                if tag not in tag_to_ix: tag_to_ix[tag] = len(tag_to_ix)
                    
        return [x1_to_ix, x2_to_ix, x3_to_ix, tag_to_ix]

def reduce_vocabs(dataset, _to_ix, min_freq):

    if min_freq != None:
        for i in range(len(min_freq)): # exclude tag_to_ix
            if min_freq[i] > 1:
                vocab = [x for data in dataset for x in data[i]]
                sc = Counter(vocab)

                newx_to_ix = {Const.UNK_TOKEN: Const.UNK_ID, 
                             Const.PAD_TOKEN: Const.PAD_ID, 
                             Const.BOS_TOKEN: Const.BOS_ID, 
                             Const.EOS_TOKEN: Const.EOS_ID}
                
                for k,v in _to_ix[i].items():
                    if sc[k] >= min_freq[i] and k not in newx_to_ix:
                        newx_to_ix[k] = len(newx_to_ix)

                _to_ix[i] = newx_to_ix

    return _to_ix

def addsep(obj):
    '''
    add separation between symbols so we can separate them from characters
    we want to separate symbols from characters to improve the accuracy of syllable_tokenizer
    '''
    obj = re.sub(r'([^\|])([-_/ !"#%&,:;<=>@~0-9๐๑๒๓๔๕๖๗๘๙‘’\ufeff\r\t\\\^\$\[\]\(\)\{\}\.\?\*\+\'])', r'\1|\2', obj)
    obj = re.sub(r'([^\|])([-_/ !"#%&,:;<=>@~0-9๐๑๒๓๔๕๖๗๘๙‘’\ufeff\r\t\\\^\$\[\]\(\)\{\}\.\?\*\+\'])', r'\1|\2', obj) # twice
    obj = re.sub(r'([-_/ !"#%&,:;<=>@~0-9๐๑๒๓๔๕๖๗๘๙‘’\ufeff\r\t\\\^\$\[\]\(\)\{\}\.\?\*\+\'])([^\|])', r'\1|\2', obj)
    obj = re.sub(r'([-_/ !"#%&,:;<=>@~0-9๐๑๒๓๔๕๖๗๘๙‘’\ufeff\r\t\\\^\$\[\]\(\)\{\}\.\?\*\+\'])([^\|])', r'\1|\2', obj) # twice
    obj = re.sub(r'\|{2,5}', '|', obj)
    return obj

def build_dataset(data_name, charlevel=True, min_freq=None):
    '''
    for code to be simple, test will have dummy labels
    '''
    for directory in ['train', 'validate', 'test']:
        
        dataset = []
        pat = re.compile('<[^>]{0,6}>')
        filenames = os.listdir('data/' + directory)
        onetenth = max(1, len(filenames)//10) # for tracking progress
        for i, filename in enumerate(filenames):
            
            with open ('data/' + directory + '/' + filename) as file: 
                obj = file.read()
                obj = re.sub(pat, '', obj)
                lines = obj.split('\n')

            for line in lines:
                words = line.strip('|').split('|')
                syllables_ofline, chars_ofline, labels_ofline = [], [], []
                
                for word in words: # word is cut as correct group of syllables
                    wordwithsep = addsep(word) # separate symbols from characters within each word before syllable_tokenize
                    parts = wordwithsep.strip('|').split('|')
                    syllables_ofword, chars_ofword = [], []
                    
                    # build syllables_ofword
                    for part in parts:
                        syllables_ofpart = stok(part)
                        syllables_ofword += syllables_ofpart
                        
                    # for syllable model, build syllable_ofword for each line
                    syllables_ofline += syllables_ofword
                        
                    # for char model, build chars_ofword for each line
                    if charlevel==True:
                        
                        chars_ofword = list(word) if word != '' else [''] # if word is '', for example in an empty line
                        chars_ofline += chars_ofword # use only in character model
                    
                    # build label_ofword for different models
                    if data_name in ['ch_1', 'bi_1', 'bict_1', 'bisy_1', 'chctbi_1', 'bictsy_1']: # scheme BI
                        
                        if len(syllables_ofword) > 0: # for sure
                            labels_ofword = ['I1'] * len(chars_ofword) # all labels is 'I1'
                            labels_ofword[0] = 'B1' # except that first labels is 'B1'
                    
                    elif data_name in ['bi_12345', 'bict_12345', 'bisy_12345']: # scheme D

                        if len(chars_ofword) < 5: 
                            labels_ofword = ['I1'] * len(chars_ofword) # all labels is 'I2'
                            labels_ofword[0] = 'B1' # # except that first labels is 'B1'

                        elif len(chars_ofword) < 10:
                            labels_ofword = ['I2'] * len(chars_ofword) # all labels is 'I2'
                            labels_ofword[0] = 'B2' # except that first labels is 'B2'

                        elif len(chars_ofword) < 15:
                            labels_ofword = ['I3'] * len(chars_ofword) # all labels is 'I3'
                            labels_ofword[0] = 'B3' # except that first labels is 'B3'

                        elif len(chars_ofword) < 20:
                            labels_ofword = ['I4'] * len(chars_ofword) # all labels is 'I4'
                            labels_ofword[0] = 'B4' # except that first labels is 'B4'
                            
                        else:
                            labels_ofword = ['I5'] * len(chars_ofword) # all labels is 'I5'
                            labels_ofword[0] = 'B5' # except that first labels is 'B5'
                    
                    elif data_name in ['bictsy_12345']: # not good, must change or delete

                        if len(syllables_ofword) == 1:
                            labels_ofword = ['I1'] * len(chars_ofword) # all labels is 'I2'
                            labels_ofword[0] = 'B1' # # except that first labels is 'B1'

                        elif len(syllables_ofword) == 2:
                            labels_ofword = ['I2'] * len(chars_ofword) # all labels is 'I2'
                            labels_ofword[0] = 'B2' # except that first labels is 'B2'

                        elif len(syllables_ofword) == 3:
                            labels_ofword = ['I3'] * len(chars_ofword) # all labels is 'I3'
                            labels_ofword[0] = 'B3' # except that first labels is 'B3'

                        elif len(syllables_ofword) == 4:
                            labels_ofword = ['I4'] * len(chars_ofword) # all labels is 'I4'
                            labels_ofword[0] = 'B4' # except that first labels is 'B4'
                            
                        else:
                            labels_ofword = ['I5'] * len(chars_ofword) # all labels is 'I5'
                            labels_ofword[0] = 'B5' # except that first labels is 'B5'
                        
                    elif data_name == 'sy_1': # scheme BI

                        if len(syllables_ofword) > 0: # for sure
                            labels_ofword = ['I1'] * len(syllables_ofword) # all labels is 'I1'
                            labels_ofword[0] = 'B1' # except that first labels is 'B1'
                            
                    elif data_name == 'sy_1-23-45': # scheme A

                        if len(syllables_ofword) < 3:
                            labels_ofword = ['I1'] * len(syllables_ofword) # all labels is 'I1'
                            labels_ofword[0] = 'B1' # except that first labels is 'B1'

                        elif len(syllables_ofword) < 5:
                            labels_ofword = ['I3'] * len(syllables_ofword) # all labels is 'I3'
                            labels_ofword[0] = 'B3' # except that first labels is 'B3'

                        else:
                            labels_ofword = ['I5'] * len(syllables_ofword) # all labels is 'I5'
                            labels_ofword[0] = 'B5' # except that first labels is 'B5'

                    elif data_name == 'sy_1234': # scheme B

                        if len(syllables_ofword) == 1:
                            labels_ofword = ['B1'] * len(syllables_ofword) # all labels is 'B1'

                        elif len(syllables_ofword) == 2:
                            labels_ofword = ['I2'] * len(syllables_ofword) # all labels is 'I2'
                            labels_ofword[0] = 'B2' # except that first labels is 'B2'

                        elif len(syllables_ofword) == 3:
                            labels_ofword = ['I3'] * len(syllables_ofword) # all labels is 'I3'
                            labels_ofword[0] = 'B3' # except that first labels is 'B3'

                        else:
                            labels_ofword = ['I4'] * len(syllables_ofword) # all labels is 'I4'
                            labels_ofword[0] = 'B4' # except that first labels is 'B4'
                            
                    elif data_name == 'sy_12345': # scheme C

                        if len(syllables_ofword) == 1:
                            labels_ofword = ['B1'] * len(syllables_ofword) # all labels is 'B1'

                        elif len(syllables_ofword) == 2:
                            labels_ofword = ['I2'] * len(syllables_ofword) # all labels is 'I2'
                            labels_ofword[0] = 'B2' # except that first labels is 'B2'

                        elif len(syllables_ofword) == 3:
                            labels_ofword = ['I3'] * len(syllables_ofword) # all labels is 'I3'
                            labels_ofword[0] = 'B3' # except that first labels is 'B3'

                        elif len(syllables_ofword) == 4:
                            labels_ofword = ['I4'] * len(syllables_ofword) # all labels is 'I4'
                            labels_ofword[0] = 'B4' # except that first labels is 'B4'
                            
                        else:
                            labels_ofword = ['I5'] * len(syllables_ofword) # all labels is 'I5'
                            labels_ofword[0] = 'B5' # except that first labels is 'B4'
                    
                    # add labels to line
                    labels_ofline += labels_ofword
                
                # for bigram model, char become bigram
                if charlevel==True:
                    chartypes_ofline = [CHAR_TYPE_FLATTEN.get(char, Const.UNK_TOKEN) for char in chars_ofline]
                    bigrams_ofline = [a+b for a,b in zip(chars_ofline, chars_ofline[1:]+[Const.EOS_TOKEN])]
                    syllables_ofline = [x for s in syllables_ofline for x in [s]*len(s)] if syllables_ofline != [''] else ['']
                    
                # add data of each line to dataset
                if data_name[:3] == 'sy_': dataset.append((syllables_ofline, labels_ofline))
                elif data_name in ['ch_1']: dataset.append((chars_ofline, labels_ofline))
                elif data_name in ['bi_1', 'bi_12345']: dataset.append((bigrams_ofline, labels_ofline))
                elif data_name in ['bict_1', 'bict_12345']: dataset.append((bigrams_ofline, chartypes_ofline, labels_ofline))
                elif data_name in ['bisy_1', 'bisy_12345']: dataset.append((bigrams_ofline, syllables_ofline, labels_ofline))
                elif data_name in ['chctbi_1']: dataset.append((chars_ofline, chartypes_ofline, bigrams_ofline, labels_ofline))
                elif data_name in ['bictsy_1', 'bictsy_12345']: dataset.append((bigrams_ofline, chartypes_ofline, syllables_ofline, labels_ofline))
                
            show_progress(i, onetenth) # for tracking progress
                
        alldataset.append(dataset)
        print('', directory, 'length =', len(dataset))

        if directory == 'train':
            
            _to_ix = get_vocabs(dataset)
            print('size of dict:', [(i, len(_to_ix[i])) for i in range(len(_to_ix))])
            
            _to_ix = reduce_vocabs(dataset, _to_ix, min_freq)
            print('size of reduced dict:', [(i, len(_to_ix[i])) for i in range(len(_to_ix))])

            dataset.append(_to_ix)
    
    with open('dataset/' + data_name + '.json', 'w') as file:
            json.dump(dataset, file)