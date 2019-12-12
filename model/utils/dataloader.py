import numpy as np
import random
import torch
import collections
import pickle
from torch.utils.data import DataLoader, Dataset


def file_sort(logfile):
    nfiles = 0
    file_cnt = collections.defaultdict(int)
    with open(logfile) as log:
        for line in log:
            line = line.split('|')
            timestampe = int(line[3])
            filename = line[5].strip('\n')
            file_cnt[filename] += 1
            nfiles += 1
    sorted_file_cnt = []
    for f in sorted(file_cnt, key=file_cnt.get):
        sorted_file_cnt.append((f, file_cnt[f]))
    return sorted_file_cnt, nfiles


def path_merge(file_cnt, t=5):
    file_merged = collections.defaultdict(int)
    for file, count in file_cnt:
        if count < t:
            file = file.split('\\')
            file = '\\'.join(file[:-1]) + '\\X'
        file_merged[file] += count 
    return file_merged


def load_data(logfile, ratio=0.8):
    path_list = []
    time_list = []
    with open(logfile) as log:
        for line in log:
            line = line.split('|')
            time_list.append(int(line[3]))
            path_list.append(line[5].strip('\n'))
    if ratio is None:
        return path_list, time_list, None, None
    else:
        train_size = int(len(path_list)*ratio)
        return path_list[:train_size], time_list[:train_size], path_list[train_size:], time_list[:train_size]


def get_test_data(pos_list, pos_time, neg_lists=[], neg_times=[], time_interval=60):
    start_time = None
    start_idx = 0
    for i in range(len(pos_time)):
        if start_time is None:
            start_time = pos_time[i]
            start_idx = i
        elif pos_time[i] - start_time >= time_interval:
            yield pos_list[start_idx:i], pos_time[start_idx:i], 0
            start_time = pos_time[i]
    if start_idx < len(pos_time):
        yield pos_list[start_idx:], pos_time[start_idx:], 0
    for neg_list, neg_time in zip(neg_lists, neg_times):
        start_time = None
        start_idx = 0
        for i in range(len(neg_time)):
            if start_time is None:
                start_time = neg_time[i]
                start_idx = i
            elif neg_time[i] - start_time >= time_interval:
                yield neg_list[start_idx:i], neg_time[start_idx:i], 1
                start_time = pos_time[i]
        if start_idx < len(neg_time):
            yield neg_list[start_idx:], neg_time[start_idx:], 1


def get_batch(path_list, time_list, time_window, seq_windows, batch_size, print_step=1000, sample_ratio=1.0):
    # path_pairs = []
    batch = []
    label = []
    step = 0
    for i in range(len(path_list)):
        for j in range(i-1, max(-1, i-seq_windows-1), -1):
            if(time_list[i] - time_list[j] < time_window):
                # path_pairs.append((path_list[i][1], path_list[j][1]))
                batch.append(path_list[i])
                label.append(path_list[j])
            else:
                break
        for j in range(i+1, min(i+seq_windows+1, len(path_list))):
            if(time_list[j] - time_list[i] < time_window):
                # path_pairs.append((path_list[i][1], path_list[j][1]))
                batch.append(path_list[i])
                label.append(path_list[j])
            else:
                break
        if(len(batch) >= batch_size):
            step += 1
            if print_step is not None and step % print_step == 0:
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


class FileDictionary:
    def __init__(self):
        self.t = 0.0001
        self.file_cnt = collections.defaultdict(int)
        self.dicts = collections.defaultdict(int)
        self.subdicts = collections.defaultdict(int)
        self.subpaths = []
        self.pdiscard = []
    
    def fit(self, file_list, threshold=5):
        for file in file_list:
            self.file_cnt[file] += 1
        self.threshold(threshold)
        self.dicts = {f:i for i, f in enumerate(self.file_cnt.keys())}
        self.initDiscard()
        self.initSubpath()

    def maxdepth(self):
        max_depth = 0
        for file in self.file_cnt:
            max_depth = max([max_depth, len(file.split('\\'))])
        return max_depth

    def threshold(self, t):
        max_depth = self.maxdepth()
        file_cnt_ = collections.defaultdict(int)
        for file in self.file_cnt:
            if self.file_cnt[file] < t:
                file_cnt_[file] = self.file_cnt[file]
        for file in file_cnt_:
            del self.file_cnt[file]
            
        for depth in range(max_depth, 1, -1):
            file_cnt_1 = collections.defaultdict(int)
            file_cnt_2 = collections.defaultdict(int)
            for file in file_cnt_:
                file_depth = len(file.split('\\'))
                if file_depth < depth:
                    file_cnt_1[file] = file_cnt_[file]
                else:
                    file_ = '\\'.join(file.split('\\')[:depth-1]) + '\\X'
                    file_cnt_2[file_] += file_cnt_[file]
            if len(file_cnt_2) == 0: continue
            for file in file_cnt_2:
                if file_cnt_2[file] >= t:
                    self.file_cnt[file] = file_cnt_2[file]
                else:
                    file_cnt_1[file] = file_cnt_2[file]
            file_cnt_ = file_cnt_1

    def initDiscard(self):
        ntoken = 0
        for file in self.file_cnt:
            ntoken += self.file_cnt[file]
        for file in self.dicts:
            p = self.file_cnt[file] / ntoken
            self.pdiscard.append((self.t / p ) ** 0.5 + self.t / p)

    def getSubPath(self, path:str) -> list:
        subpath = []
        path = path.split('\\')
        for i in range(1, len(path)):
            subpath.append('S' + '\\'.join(path[:i]))
        return subpath

    def initSubpath(self):
        idx = len(self.dicts)
        for file in self.dicts:
            self.subpaths.append([])
            subpaths = self.getSubPath(file)
            for subpath in subpaths:
                if subpath not in self.subdicts:
                    self.subdicts[subpath] = idx
                    idx += 1
                self.subpaths[-1].append(self.subdicts[subpath])

    def discard(self, file_idx:int) -> bool:
        assert(file_idx >= 0)
        assert(file_idx < len(self.dicts))
        return random.uniform(0, 1) > self.pdiscard[file_idx]
    
    def transform(self, file_list:list) -> list:
        trans_list = []
        for file in file_list:
            if file in self.dicts:
                trans_list.append(self.dicts[file])
                continue
            file = file.split('\\')
            for i in range(len(file)-1, 0, -1):
                path = '\\'.join(file[:i]) + '\\X'
                if path in self.dicts:
                    trans_list.append(self.dicts[path])
                    break
        # In current dataset, this assert always success
        assert(len(file_list) == len(trans_list))
        return trans_list

    def unique(self):
        return self.dicts.keys()

    def size(self):
        return len(self.dicts)

    def subpathsize(self):
        return len(self.subdicts)

    def save(self, targetfile):
        with open(targetfile, 'wb') as f:
            pickle.dump(self.dicts, f)
            pickle.dump(self.subdicts, f)
            pickle.dump(self.subpaths, f)
            pickle.dump(self.pdiscard, f)

    def load(self, targetfile):
        with open(targetfile, 'rb') as f:
            self.dicts = pickle.load(f)
            self.subdicts = pickle.load(f)
            self.subpaths = pickle.load(f)
            self.pdiscard = pickle.load(f)


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
    train_data, train_time, valid_data, _ = load_data(train_path, ratio=0.8)
    print('Length of train data: %d' % len(train_data))
    dictionary = FileDictionary()
    dictionary.fit(train_data)
    cnt = 0
    for file, p in zip(dictionary.file_cnt, dictionary.pdiscard):
        if p < 1:
            cnt += (dictionary.file_cnt[file] * (1-p))
    print(cnt)
