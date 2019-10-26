import pickle
import time

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from utils import dataloader, plot


class File:
    def __init__(self, size, i):
        self.name = '/'
        self.i = i
        self.parent = None
        self.files = {}
        self.w = 0.5
        self.b = [0.0] * size


class FileSystem:
    def __init__(self, vec_size):
        self.inodes = []
        self.size = vec_size
        self.inodes.append(File(vec_size, len(self.inodes)))

    def addFile(self, path):
        path = path.split('\\')
        node = self.inodes[0]
        for subpath in path:
            if subpath == '':
                continue
            if subpath in node.files:
                node = node.files[subpath]
            else:
                self.inodes.append(File(self.size, len(self.inodes)))
                self.inodes[-1].name = subpath
                self.inodes[-1].parent = node.i
                node.files[subpath] = self.inodes[-1]
                node = self.inodes[-1]

    def getFile(self, path):
        path = path.split('\\')
        node = self.inodes[0]
        for subpath in path:
            if subpath == '':
                continue
            if subpath in node.files:
                node = node.files[subpath]
            else:
                return None
        return node

    def sumVec(self, File):
        return [0.1] * self.size

    def getPathInput(self, path):
        idx = np.array([0.0] * len(self.inodes))
        node = self.inodes[0]
        idx[0] = 1
        path = path.split('\\')
        for subpath in path:
            if subpath == '':
                continue
            if subpath in node.files:
                node = node.files[subpath]
                idx = idx*node.w
                idx[node.i] += 1
        return idx

    def getVecTable(self):
        return np.array([node.b for node in self.inodes])


def read_file_access_log(logfile):
    with open(logfile) as log:
        for line in log:
            line = line.split('|')
            yield line[5]


def path2vec(target_log='intermediate/User11.log'):
    vec_size = 10
    time_window = 5
    fs = FileSystem(vec_size)
    for r in read_file_access_log(target_log):
        fs.addFile(r)
    corpus_size = len(fs.inodes)
    print('corpus size is: ', corpus_size)

    W1 = Variable(torch.randn(vec_size, corpus_size).float(),
                  requires_grad=True)
    W2 = Variable(torch.randn(corpus_size, vec_size).float(),
                  requires_grad=True)
    num_epochs = 1
    learning_rate = 0.001
    batch_size = 10000

    for epo in range(num_epochs):
        for pairs in dataloader.get_batch(target_log, time_window, 40, batch_size):
            loss_val = 0
            for center, context in pairs:
                x = Variable(torch.from_numpy(fs.getPathInput(center))
                             ).float()  # more optimization need
                y = Variable(torch.from_numpy(
                    np.array([fs.getFile(context).i])).long())

                z1 = torch.matmul(W1, x)
                z2 = torch.matmul(W2, z1)
                log_softmax = F.log_softmax(z2, dim=0)

                loss = F.nll_loss(log_softmax.view(1, -1), y)
                loss_val += loss.data.item()
                loss.backward()
                W1.data -= learning_rate * W1.grad.data
                W2.data -= learning_rate * W2.grad.data

                W1.grad.data.zero_()
                W2.grad.data.zero_()
            print(f'Loss for batch : {loss_val/batch_size}')


class skip_gram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(skip_gram, self).__init__()
        self.v_emb = nn.Embedding(vocab_size, embedding_dim)
        self.u_emb = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.log_sigmoid = nn.LogSigmoid()

        # init emb param, Xavier init
        initrange = (2.0 / (vocab_size + embedding_dim)) ** 0.5
        self.v_emb.weight.data.uniform_(-initrange, initrange)
        self.u_emb.weight.data.uniform_(-0, 0)

    def forward(self, v_pos, u_pos, neg_pos):
        # v,u: [batch_size, embedding_dim]
        v = self.v_emb(v_pos)
        u = self.u_emb(u_pos)
        
        # positive_val: [batch_size]
        positive_val = self.log_sigmoid(torch.sum(v*u, dim=1)).squeeze()

        neg = self.u_emb(neg_pos)
        # neg_val: [batch_size, neg_size]
        neg_val = torch.bmm(neg, v.unsqueeze(2)).squeeze(2)
        # negative_val: [batch_size]
        negative_val = self.log_sigmoid(-torch.sum(neg_val, dim=1)).squeeze()
        
        # 使用负采样的一个问题：如何计算异常
        # 可能的topN的相似度
        loss = positive_val + negative_val
        return -loss.mean()

def save_model(model, outpath):
    torch.save(model.state_dict(), outpath+'/model.pt')

def load_model(model, modelpath):
    torch.load_state_dict(torch.load(modelpath))

def _path2vec(target_log='intermediate/User11.log', load_epoch=0):
    emb_dim = 20
    time_window = 5
    seq_window = 40

    file_to_idx = {}
    for r in read_file_access_log(target_log):
        if r not in file_to_idx:
            file_to_idx[r] = len(file_to_idx)
    corpus_size = len(file_to_idx)

    num_epochs = 5
    learning_rate = 1.0
    batch_size = 256
    log_step=100

    train_data, test_data = dataloader.read_data(target_log, ratio=0.1)
    unigram_table = []
    for r in train_data:
        r[1] = file_to_idx[r[1]]
        unigram_table.append(r[1])
    model = skip_gram(corpus_size, emb_dim)
    optim = SGD(model.parameters(), lr=learning_rate)
    if(load_epoch != 0):
        model.load_state_dict(torch.load('model/runs/path2vec_epoch%d.pt'%load_epoch))

    writer = SummaryWriter('model/runs/path2vec')

    step = 0
    for epo in range(num_epochs):
        avg_loss = 0
        start_time = time.time()
        for batch, label in dataloader.get_batch(train_data, time_window, seq_window, batch_size, print_step=log_step):
            batch_neg = dataloader.get_neg_data(unigram_table, 10, batch_size, batch)

            batch_input = torch.tensor(batch, dtype=torch.long)
            batch_label = torch.tensor(label, dtype=torch.long)
            batch_neg= torch.tensor(batch_neg, dtype=torch.long)

            loss = model(batch_input, batch_label, batch_neg)
            optim.zero_grad()
            loss.backward()
            optim.step()

            step += 1
            avg_loss += loss.item()
            if step % log_step == 0:
                print('Average loss at step %d: %f' % (step, avg_loss/log_step))
                writer.add_scalar('training loss', avg_loss/log_step, step)
                avg_loss = 0
        print('eopch time cost: %d s' % (time.time()-start_time))
        start_time = time.time()

    torch.save(model.state_dict(), 'model/runs/path2vec_epoch%d.pt'%(num_epochs+load_epoch))

###
def gen_neigh_set(data):
    neigh = {}
    for batch, label in dataloader.get_batch(data, 5, 40, 10000):
        for center, context in zip(batch, label):
            if(center not in neigh):
                neigh[center] = set()
            neigh[center].add(context)
    return neigh

def save_neigh(neigh, outfile):
    with open(outfile, 'wb') as of:
        pickle.dump(neigh, of, pickle.HIGHEST_PROTOCOL)

def load_neigh(file):
    with open(file, 'rb') as infile:
        return pickle.load(infile)

def score(train_neigh, neigh, debug=0):
    score = 0
    scores = []
    for file in neigh:
        if file in train_neigh:
            union = train_neigh[file] & neigh[file]
            s = len(union) / len(neigh[file])
            score += s
            scores.append(s)

            if(s < debug):
                print('Low score for file: %sTrain neigh: %s\n\nRun neigh: %s\n\n'
                    % (file, str(train_neigh[file]), str(neigh[file])))
        # else:
        #     scores.append(0)
    return score / len(neigh), scores

def _neigh():
    train_data, test_data = dataloader.read_data('intermediate/User11.log', ratio=0.8)
    # train_neigh = gen_neigh_set(train_data)
    # save_neigh(train_neigh, 'intermediate/train_neigh.pickle')
    # test_neigh = gen_neigh_set(test_data)
    # save_neigh(test_neigh, 'intermediate/test_neigh.pickle')
    train_neigh = load_neigh('intermediate/train_neigh.pickle')
    test_neigh = load_neigh('intermediate/test_neigh.pickle')

    attack1_data, _ = dataloader.read_data('intermediate/User11Attack1.log', ratio =1.0)
    attack1_neigh = gen_neigh_set(attack1_data)
    attack2_data, _ = dataloader.read_data('intermediate/User11Attack2.log', ratio =1.0)
    attack2_neigh = gen_neigh_set(attack2_data)    
    attack3_data, _ = dataloader.read_data('intermediate/User11Attack3.log', ratio =1.0)
    attack3_neigh = gen_neigh_set(attack3_data)

    fig = go.Figure()
    y_true = []
    y_score = []
    _, scores = score(train_neigh, test_neigh)
    y_true = y_true + [1] * len(scores)
    y_score += scores
    # plot.plot_score(fig, scores, name='Test')
    _, scores = score(train_neigh, attack1_neigh)
    y_true = y_true + [0] * len(scores)
    y_score += scores
    # plot.plot_score(fig, scores, name='Attack1')
    _, scores = score(train_neigh, attack2_neigh)
    y_true = y_true + [0] * len(scores)
    y_score += scores
    # plot.plot_score(fig, scores, name='Attack2')
    _, scores = score(train_neigh, attack3_neigh)
    y_true = y_true + [0] * len(scores)
    y_score += scores
    # plot.plot_score(fig, scores, name='Attack3')
    plot.plot_roc(fig, y_true, y_score)
    fig.show()

if __name__ == "__main__":
    _path2vec(load_epoch=0)
