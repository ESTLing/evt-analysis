import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset


def read_data(logfile, ratio=0.8):
    path_list = []
    with open(logfile) as log:
        for line in log:
            line = line.split('|')
            path_list.append([int(line[3]), line[5]])

    train_size = int(len(path_list)*ratio)
    return path_list[:train_size], path_list[train_size:]


def get_batch(path_list, time_window, seq_windows, batch_size, print_step=1000):
    # path_pairs = []
    batch = []
    label = []
    step = 0
    for i in range(len(path_list)):
        for j in range(i-1, max(-1, i-seq_windows-1), -1):
            if(path_list[i][0] - path_list[j][0] < time_window):
                # path_pairs.append((path_list[i][1], path_list[j][1]))
                batch.append(path_list[i][1])
                label.append(path_list[j][1])
            else:
                break
        for j in range(i+1, min(i+seq_windows+1, len(path_list))):
            if(path_list[j][0] - path_list[i][0] < time_window):
                # path_pairs.append((path_list[i][1], path_list[j][1]))
                batch.append(path_list[i][1])
                label.append(path_list[j][1])
            else:
                break
        if(len(batch) >= batch_size):
            step += 1
            if step % print_step == 0:
                print('Path Porcess: %d/%d' % (i, len(path_list)), end='\t')
            # yield np.array(path_pairs[:batch_size])
            yield np.array(batch[:batch_size]), np.array(label[:batch_size])
            # path_pairs = path_pairs[batch_size:]
            batch = batch[batch_size:]
            label = label[batch_size:]

    return np.array(batch), np.array(label)


def get_neg_data(unigram_table, num, batch_size, batch):
    neg = np.zeros((num))
    for i in range(batch_size):
        delta = random.sample(unigram_table, num)
        while batch[i] in delta:
            delta = random.sample(unigram_table, num)
        neg = np.vstack([neg, delta])
    return neg[1:batch_size+1]


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


class FSDataset(Dataset):
    ''' Dataset class for read fs data
    '''

    def __init__(self, fs, file_path, max_time_window=30, max_seq_size=100):
        self.event_seqs = []
        self.time_seqs = []

        with open(file_path) as log:
            seq = []
            for i, line in enumerate(log):
                line = line.split('|')
                i = fs.getFile(line[5]).i
                if len(seq) == max_seq_size or \
                        (len(seq) > 0 and int(line[3]) - int(seq[0][0]) > max_time_window):
                    self.event_seqs.append(torch.IntTensor(
                        [int(event[1]) for event in seq]))
                    self.time_seqs.append(torch.IntTensor(
                        [int(event[0]) for event in seq]))
                    seq.clear()
                seq.append((line[3], i))
            self.event_seqs.append(torch.IntTensor(
                [int(event[1]) for event in seq]))
            self.time_seqs.append(torch.IntTensor(
                [int(event[0]) for event in seq]))

    def __len__(self):
        return len(self.event_seqs)

    def __getitem__(self, index):
        return self.event_seqs[index]


def buildFS(file_path, hidden_size=100):
    fs = FileSystem(hidden_size)
    with open(file_path) as log:
        for line in log:
            line = line.split('|')
            fs.addFile(line[5])
    return fs


def pad_batch_fn(batch_data):
    sorted_batch = sorted(batch_data, key=lambda x: x.size(), reverse=True)
    sorted_batch = [seq.int() for seq in sorted_batch]
    seqs_length = torch.IntTensor(list(map(len, sorted_batch)))

    event_seq_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).int()

    for idx, (event_seq, seq_len) in enumerate(zip(sorted_batch, seqs_length)):
        event_seq_tensor[idx, :seq_len] = torch.IntTensor(event_seq)
    return event_seq_tensor, seqs_length


if __name__ == "__main__":
    train_path = 'intermediate/User11.log'
    fs = buildFS(train_path)
    print('Corpus size: %d' % len(fs.inodes))
    dataset = FSDataset(fs, train_path)
    dataloader = DataLoader(dataset, batch_size=32,
                            collate_fn=pad_batch_fn, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(i, batch[0].shape)
