from pathlib import Path
from model import Const

# get pred and correct tags (without padding) for computing accuracy
def get_tags(seqs, xtags, lengths):
    pred_tags = [seq[:length] for seq, length in zip(seqs, lengths)] 
    tags = [xtag[:length] for xtag, length in zip(xtags.tolist(), lengths)]
    return [pred_tags, tags]

def create_pred_text(_to_ix, test_data, pred_tags, m_type='sy'):

    tag_to_ix = _to_ix[-1]
    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
    
    # if model is bigram, we choose only the first char in bigram to build pred_text
    if m_type=='bi': 
        test_data = [([bi[0] for bi in bis], tags) for bis, tags in test_data]
    elif m_type=='chctbi': 
        test_data = [(chs, tags) for chs, _, _, tags in test_data]
    elif m_type=='bictsy': 
        test_data = [([bi[0] for bi in bis], tags) for bis, _, _, tags in test_data]

    pred_text = ''
    for i, tags in enumerate(pred_tags): # each line
        pred_text += test_data[i][0][0] # first syllable
        for j in range(1, len(tags)): # each next syllable
            if ix_to_tag[tags[j]][0] == 'B': 
                pred_text += '|' + test_data[i][0][j]
            else: 
                pred_text += test_data[i][0][j]
        pred_text += '|' + '\n' # at the end of each line
    pred_text = pred_text[:-2] + '\n' # correction at the end

    return pred_text

def evaluate(_to_ix, test_data, pred_tags, m_type):
    answer_text = Path('dataset/answer_text.txt').read_text()
    pred_text = create_pred_text(_to_ix, test_data, pred_tags, m_type)
    Path('pred_text.txt').write_text(pred_text)
    result = _compute_stats(answer_text, pred_text)
    return result

def show_progress(i, onetenth):
    if i % onetenth == onetenth - 1:print(i // onetenth, end='')


# The remaining code is from Pythainlp

def preprocessing(txt: str, remove_space: bool = True) -> str:
    
    SEPARATOR = "|"
    SURROUNDING_SEPS_RX = re.compile("{sep}? ?{sep}$".format(sep=re.escape(SEPARATOR)))
    MULTIPLE_SEPS_RX = re.compile("{sep}+".format(sep=re.escape(SEPARATOR)))
    TAG_RX = re.compile("<\/?[A-Z]+>")
    TAILING_SEP_RX = re.compile("{sep}$".format(sep=re.escape(SEPARATOR)))

    txt = re.sub(SURROUNDING_SEPS_RX, "", txt)
    if remove_space:
        txt = re.sub("\s+", "", txt)
    txt = re.sub(MULTIPLE_SEPS_RX, SEPARATOR, txt)
    txt = re.sub(TAG_RX, "", txt)
    txt = re.sub(TAILING_SEP_RX, "", txt).strip()

    return txt

import sys
import re

import numpy as np
import pandas as pd

SEPARATOR = "|"

def _f1(precision, recall):
    if precision == recall == 0:
        return 0
    return 2*precision*recall / (precision + recall)

def flatten_dict(my_dict, parent_key="", sep=":"):
    items = []
    for k, v in my_dict.items():
        new_key = "%s%s%s" % (parent_key, sep, k) if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key).items())
        else:
            items.append((new_key, v))

    return dict(items)

def benchmark(ref_samples, samples):
    results = []
    for i, (r, s) in enumerate(zip(ref_samples, samples)):
        try:
            r, s = preprocessing(r), preprocessing(s)
            if r and s:
                stats = _compute_stats(r, s)
                stats = flatten_dict(stats)
                stats["expected"] = r
                stats["actual"] = s
                results.append(stats)
        except:
            reason = """
                [Error]
                Reason: %s
                Pair (i=%d)
                --- label
                %s
                --- sample
                %s
                """ % (sys.exc_info(), i, r, s)

            raise SystemExit(reason)

    return pd.DataFrame(results)

def preprocessing(sample, remove_space=True):
    # prevent tailing separator and <NE></NE> tag
    sample = re.sub(
        re.compile("{sep}? ?{sep}$".format(sep=re.escape(SEPARATOR))),
        "",
        sample
    )

    sample = re.sub(
        re.compile("^{sep}? ?{sep}".format(sep=re.escape(SEPARATOR))),
        "",
        sample
    )

    if remove_space:
        sample = re.sub(
            "\s+",
            "",
            sample
        )

    sample = re.sub(
        re.compile("{sep}+".format(sep=re.escape(SEPARATOR))),
        SEPARATOR,
        sample
    )

    sample = re.sub(r"<\/?[A-Z]+>", "", sample)

    sample = re.sub(
        re.compile("{sep}$".format(sep=re.escape(SEPARATOR))),
        "",
        sample
    ).strip()

    return sample

def _compute_stats(ref_sample, raw_sample):
    ref_sample, _ = _binary_representation(ref_sample)
    sample, _ = _binary_representation(raw_sample)

    # Charater Level
    c_pos_pred, c_neg_pred = np.argwhere(sample==1), np.argwhere(sample==0)

    c_pos_pred = c_pos_pred[c_pos_pred < ref_sample.shape[0]]
    c_neg_pred = c_neg_pred[c_neg_pred < ref_sample.shape[0]]

    c_tp = np.sum(ref_sample[c_pos_pred] == 1)
    c_fp = np.sum(ref_sample[c_pos_pred] == 0)

    c_tn = np.sum(ref_sample[c_neg_pred] == 0)
    c_fn = np.sum(ref_sample[c_neg_pred] == 1)

    c_precision = c_tp / (c_tp + c_fp)
    c_recall = c_tp / (c_tp + c_fn)
    c_f1 = _f1(c_precision, c_recall)

    # Word Level
    word_boundaries = _find_word_boudaries(ref_sample)

    correctly_tokenised_words = _count_correctly_tokenised_words(
        sample,
        word_boundaries
    )

    w_precision = correctly_tokenised_words / np.sum(sample)
    w_recall = correctly_tokenised_words / np.sum(ref_sample)
    w_f1 = _f1(w_precision, w_recall)


    ss_boundaries = _find_word_boudaries(sample)
    tokenisation_indicators = _find_words_correctly_tokenised(
        word_boundaries,
        ss_boundaries
    )

    tokenisation_indicators = list(map(lambda x: str(x), tokenisation_indicators))

    return {
        'char_level': {
            'tp': c_tp,
            'fp': c_fp,
            'tn': c_tn,
            'fn': c_fn,
            'precision': c_precision,
            'recall': c_recall,
            'f1': c_f1
        },
        'word_level': {
            'precision':  w_precision,
            'recall':  w_recall,
            'f1': w_f1
        },
        'global': {
            'tokenisation_indicators': "".join(tokenisation_indicators)
        }
    }

"""
ผม|ไม่|ชอบ|กิน|ผัก -> 10100...
"""
def _binary_representation(sample, verbose=False):
    chars = np.array(list(sample))

    boundary = np.argwhere(chars == SEPARATOR).reshape(-1)
    boundary = boundary - np.array(range(boundary.shape[0]))

    bin_rept = np.zeros(len(sample) - boundary.shape[0])
    bin_rept[list(boundary) + [0]] = 1

    sample_wo_seps = list(sample.replace(SEPARATOR, ""))
    assert len(sample_wo_seps) == len(bin_rept)

    if verbose:
        for c, m in zip(sample_wo_seps, bin_rept):
            print('%s -- %d' % (c, m))

    return bin_rept, sample_wo_seps

"""
sample: a binary representation
return array of (start, stop) indicating starting and ending position of each word
"""
def _find_word_boudaries(sample):
    boundary = np.argwhere(sample == 1).reshape(-1)
    start_idx = boundary
    stop_idx = boundary[1:].tolist() + [sample.shape[0]]

    return list(zip(start_idx, stop_idx))

"""
sample: a binary representation
word_boundaries: [ (start, stop), ... ]
"""
def _count_correctly_tokenised_words(sample, word_boundaries):
    count = 0
    for st, end in word_boundaries:
        pend = min(end, sample.shape[0])
        if ( sample[st] == 1 and np.sum(sample[st+1:pend]) == 0 ) \
            and (
                ( pend == sample.shape[0] ) or
                ( pend != sample.shape[0] and sample[pend] == 1 )
            ):
            count = count + 1

    return count

def _find_words_correctly_tokenised(ref_boundaries, predicted_boundaries):
    ref_b = dict(zip(ref_boundaries, [1]*len(ref_boundaries)))

    labels = tuple(map(lambda x: ref_b.get(x, 0), predicted_boundaries))
    return labels    