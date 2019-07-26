import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

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
            if subpath == '': continue
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
            if subpath == '': continue
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
            if subpath == '': continue
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

def get_file_pairs(logfile, time_window, batch_size):
    path_list = []
    with open(logfile) as log:
        for line in log:
            line = line.split('|')
            if(len(path_list) == 0 or line[5] != path_list[-1][1]):
                path_list.append((int(line[3]), line[5]))

    path_pairs = []
    for i in range(len(path_list)):
        for j in range(i-1, -1, -1):
            if(path_list[i][0] - path_list[j][0] < time_window):
                path_pairs.append((path_list[i][1], path_list[j][1]))
            else:
                break
        for j in range(i+1, len(path_list)):
            if(path_list[j][0] - path_list[i][0] < time_window):
                path_pairs.append((path_list[i][1], path_list[j][1]))
            else:
                break
        if(len(path_pairs) >= batch_size):
            yield np.array(path_pairs[:batch_size])
            path_pairs = path_pairs[batch_size:]
            print('Path read: %d/%d' % (i, len(path_list)))
    return np.array(path_pairs)

target_log = 'intermediate/User11.log'
vec_size=10
time_window = 5
fs = FileSystem(vec_size)
for r in read_file_access_log(target_log):
    fs.addFile(r)
corpus_size = len(fs.inodes)
print('corpus size is: ', corpus_size)

W1 = Variable(torch.randn(vec_size, corpus_size).float(), requires_grad=True)
W2 = Variable(torch.randn(corpus_size, vec_size).float(), requires_grad=True)
num_epochs = 1
learning_rate = 0.001
batch_size = 10000

for epo in range(num_epochs):
    for pairs in get_file_pairs(target_log, time_window, batch_size):
        loss_val = 0
        for center, context in pairs:
            x = Variable(torch.from_numpy(fs.getPathInput(center))).float() # more optimization need
            y = Variable(torch.from_numpy(np.array([fs.getFile(context).i])).long())

            z1 = torch.matmul(W1, x)
            z2 = torch.matmul(W2, z1)
            log_softmax = F.log_softmax(z2, dim=0)

            loss = F.nll_loss(log_softmax.view(1,-1), y)
            loss_val += loss.data.item()
            loss.backward()
            W1.data -= learning_rate * W1.grad.data
            W2.data -= learning_rate * W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_() 
        print(f'Loss for batch : {loss_val/batch_size}')