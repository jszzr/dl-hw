import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
# from pyciderevalcap.ciderD.ciderD import CiderD
from pycocoevalcap.bleu.bleu import Bleu

Bleu_scorer = Bleu(4)

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, imgs, gen_result, caps, sample_size, beam_k=1, max_len=120, temperature=1.0):
    batch_size = imgs.size(0)
    seq_per_img = batch_size // sample_size

    # Generate sample captions using sampling
    model.eval()
    with torch.no_grad():
        gen_texts = model.sample(imgs, beam_k=beam_k, max_len=max_len, temperature=temperature)

    # Generate greedy baseline captions
    
    greedy_texts = model.generate_by_beamsearch(imgs, beam_k=1)

    model.train()

    # gen_result = nn.utils.rnn.pad_sequence([torch.tensor(sublist) for sublist in gen_result], batch_first=True, padding_value=-1)
    
    greedy_result = nn.utils.rnn.pad_sequence([torch.tensor(sublist) for sublist in greedy_texts], batch_first=True, padding_value=-1)

    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_result[i])]

    gts = OrderedDict()
    for i in range(sample_size):
        gts[i] = [array_to_str(caps[i * seq_per_img])]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
    bleu_scores = np.array(bleu_scores[3])
    print('Bleu scores:', _[3])

    scores = bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        '''
        This function computes
            log(y_t) * reward * mask_t  (where mask_t zeroes out non-words in the sequence)
        given
            input = predicted probability
            sequence = predicted word index
            reward = ...
        '''

        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output