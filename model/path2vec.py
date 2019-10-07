import numpy as np
import torch
import pickle
import plotly.graph_objects as go
from torch.autograd import Variable
from torch.nn import functional as F

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
        for pairs in dataloader.get_file_pairs(target_log, time_window, 40, batch_size):
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


###
def gen_neigh_set(data):
    neigh = {}
    for pairs in dataloader.get_file_pairs(data, 5, 40, 10000):
        for center, context in pairs:
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

if __name__ == "__main__":
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
    _, scores = score(train_neigh, test_neigh)
    plot.plot_score(fig, scores, name='Test')
    _, scores = score(train_neigh, attack1_neigh)
    plot.plot_score(fig, scores, name='Attack1')
    _, scores = score(train_neigh, attack2_neigh)
    plot.plot_score(fig, scores, name='Attack2')
    _, scores = score(train_neigh, attack3_neigh)
    plot.plot_score(fig, scores, name='Attack3')
    fig.show()