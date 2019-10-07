import numpy as np
from sklearn import metrics
import plotly.graph_objects as go

from utils import plot

def idx_item(seq):
    item2idx = {}
    idx2item = []
    for i in range(len(seq)):
        if(seq[i] not in item2idx):
            item2idx[seq[i]] = len(item2idx)
            idx2item.append(seq[i])
    return item2idx, idx2item

def ngram(seq, item2idx, n=2):
    grams = []
    for i in range(len(seq)-n+1):
        grams.append(str(item2idx.get(seq[i], 'unknown')) + ":" + str(item2idx.get(seq[i+1], 'unknown')))
    return grams

def load_data(file_path):
    seq = []
    with open(file_path) as log:
        for line in log:
            line = line.split('|')
            seq.append(line[5])
    return seq

def state_transition(grams):
    transition_prob = {}
    for i in range(len(grams)-1):
        if(grams[i] not in transition_prob):
            transition_prob[grams[i]] = {}
            transition_prob[grams[i]]['sum'] = 0
        if(grams[i+1] not in transition_prob[grams[i]]):
            transition_prob[grams[i]][grams[i+1]] = 0
        transition_prob[grams[i]][grams[i+1]] += 1
        transition_prob[grams[i]]['sum'] += 1
    for i in transition_prob:
        probs = transition_prob[i]
        for key in probs:
            if(key != 'sum'):
                probs[key] /= probs['sum']
        del probs['sum']
    return transition_prob

def separate_seq(seq, window_size=100):
    splited_seq = []
    idx = 0
    while(idx < len(seq)):
        if(len(seq)-idx > 2):
            splited_seq.append(seq[idx:idx+window_size])
        idx += window_size
    return splited_seq

def evluation(seq, item2idx, transition_prob, n=2):
    _grams = ngram(seq, item2idx, n)
    y = 0
    for i in range(len(_grams)-1):
        if(_grams[i] in transition_prob and \
           _grams[i+1] in transition_prob[_grams[i]]):
            y += transition_prob[_grams[i]][_grams[i+1]]
    return y/(len(_grams)-1)

def debug_evluation(seq, item2idx, transition_prob, n=2):
    _grams = ngram(seq, item2idx, n)
    print(_grams)
    y = []
    for i in range(len(_grams)-1):
        if(_grams[i] in transition_prob and \
           _grams[i+1] in transition_prob[_grams[i]]):
            y.append(transition_prob[_grams[i]][_grams[i+1]])
        else:
            y.append(0)
    print(y)

def statis_file(item2idx, data):
    cnt = 0
    for path in data:
        if path not in item2idx:
            cnt += 1
    print('Unkown Rate: %d/%d = %f' % (cnt, len(data), cnt/len(data)))

if __name__ == "__main__":
    data = load_data('intermediate/User11.log')
    size = len(data)
    test_size = size // 5
    print("Total size: %d\t Train size: %d\t Test size: %d" % (size, size- test_size, test_size))
    train_data, test_data = data[:size-test_size], data[size-test_size:]
    
    attack1_data = load_data('intermediate/User11Attack1.log')
    attack2_data = load_data('intermediate/User11Attack2.log')
    attack3_data = load_data('intermediate/User11Attack3.log')

    item2idx, _ = idx_item(train_data)
    _2grams = ngram(train_data, item2idx)
    print("Size of corpus: %d" % len(item2idx))
    s = set(_2grams)
    print("Size of 2-gram: %d" % len(s))

    statis_file(item2idx, attack1_data)
    statis_file(item2idx, attack2_data)
    statis_file(item2idx, attack3_data)

    # Get transition probility
    transition_prob = state_transition(_2grams)
    # for i in transition_prob:
    #     print(len(transition_prob[i]), end=' ')

    y_true = []
    y_score = []
    for seq in separate_seq(test_data):
        score = evluation(seq, item2idx, transition_prob)
        y_true.append(1)
        y_score.append(score)
    for seq in separate_seq(attack1_data):
        score = evluation(seq, item2idx, transition_prob)
        y_true.append(0)
        y_score.append(score)
    # for seq in separate_seq(attack2_data):
    #     score = evluation(seq, item2idx, transition_prob)
    #     y_true.append(0)
    #     y_score.append(score)
    # for seq in separate_seq(attack3_data):
    #     score = evluation(seq, item2idx, transition_prob)
    #     y_true.append(0)
    #     y_score.append(score)
    fig = go.Figure()
    plot.plot_roc(fig, y_true, y_score)
    fig.show()

    # fig = go.Figure()
    # scores = []
    # for seq in separate_seq(test_data):
    #     score = evluation(seq, item2idx, transition_prob)
    #     scores.append(score)
    #     if(score < 0.2):
    #         debug_evluation(seq, item2idx, transition_prob)
    # plot.plot_score(fig, scores, 'test')
    # scores = []
    # for seq in separate_seq(attack1_data):
    #     score = evluation(seq, item2idx, transition_prob)
    #     scores.append(score)
    #     if(score < 0.4):
    #         debug_evluation(seq, item2idx, transition_prob)
    # plot.plot_score(fig, scores, 'attack1')
    # scores = []
    # for seq in separate_seq(attack2_data):
    #     score = evluation(seq, item2idx, transition_prob)
    #     scores.append(score)
    #     if(score < 0.2):
    #         debug_evluation(seq, item2idx, transition_prob)
    # plot.plot_score(fig, scores, 'attack2')
    # scores = []
    # for seq in separate_seq(attack3_data):
    #     score = evluation(seq, item2idx, transition_prob)
    #     scores.append(score)
    #     if(score < 0.2):
    #         debug_evluation(seq, item2idx, transition_prob)
    # plot.plot_score(fig, scores, 'attack3')
    # fig.show()