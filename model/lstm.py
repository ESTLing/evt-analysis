import torch
import torch.optim as opt
import numpy as np

from utils import dataloader

class LSTMfs_(torch.nn.Module):
    def __init__(self, fs):
        super(LSTMfs_, self).__init__()

        self.fs = fs
        self.hidden_dim = fs.size
        self.type_size = len(fs.inodes)

        # Parameters
        self.weight = torch.nn.Parameter(torch.empty(self.type_size))
        self.emb = torch.nn.Embedding(self.type_size, self.hidden_dim)
        self.lstm = torch.nn.LSTM(self.hidden_dim, self.hidden_dim)
        self.hidden2tag = torch.nn.Linear(self.hidden_dim, self.type_size)
    
    def _fs_map(self, idx_tensor):
        path = torch.zeros(len(idx_tensor), self.hidden_dim)
        for i, node in enumerate(idx_tensor):
            node = node.item()
            weight = torch.tensor(1, dtype=torch.float)
            while node is not None:
                path[i] += weight * self.emb(torch.tensor(node, dtype=torch.long))
                weight = weight * self.weight[node]
                node = self.fs.inodes[node].parent
        return path

    def forward(self, event_seqs):
        event_seqs = torch.transpose(event_seqs, 0, 1)
        batch_size = event_seqs.shape[1]
        batch_length = event_seqs.shape[0]
        
        emb_list = []
        for t in range(batch_length):
            emb_list.append(self._fs_map(event_seqs[t]))
        emb_seq = torch.stack(emb_list)

        output, (hn, cn) = self.lstm(emb_seq)
        print(output.shape)


class LSTMfs(torch.nn.Module):
    def __init__(self, fs):
        super(LSTMfs, self).__init__()

        self.fs = fs
        self.hidden_dim = fs.size
        self.type_size = len(fs.inodes)

        # Parameters
        self.weight = torch.FloatTensor(self.type_size)
        self.emb = torch.nn.Embedding(self.type_size, self.hidden_dim)
        self.gate = torch.nn.Linear(2*self.hidden_dim, 4*self.hidden_dim)
    
    def _fs_map(self, idx_tensor):
        path = torch.zeros(len(idx_tensor), self.hidden_dim)
        for i, node in enumerate(idx_tensor):
            node = node.item()
            weight = torch.FloatTensor(1)
            while node is not None:
                path[i] += weight * self.emb(torch.LongTensor(node))
                weight = weight * self.weight[node]
                node = self.fs[node].parent
        return path

    def init_states(self, batch_size):
        self.h_d = torch.zeros(batch_size, self.hidden_size)
        self.c_d = torch.zeros(batch_size, self.hidden_size)

    def forward(self, event_seqs):
        torch.transpose(event_seqs, 0, 1)
        batch_size = event_seqs.shape[1]
        batch_length = event_seqs.shape[0]

        h_list, c_list, o_list = [], [], []

        c_t = torch.zeros(batch_size, self.hidden_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim)
        for t in len(batch_length):
            emb_seq = self._fs_map(event_seqs[t])
            # feed = torch.cat((emb_seq, h), dim=1)
            (i_t,
            f_t,
            g_t,
            o_t) = torch.chunk(self.gate(torch.cat((emb_seq, h_t), dim=1)), 4, -1)

            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            h_list.append(h_t)
            c_list.append(c_t)
            o_list.append(o_t)

        h_seq = torch.stack(h_list)
        c_seq = torch.stack(c_list)
        o_seq = torch.stack(o_list)
        self.output = torch.stack((h_seq, c_seq, o_seq))
        return self.output


if __name__ == "__main__":
    train_path = 'intermediate/User11.log'
    fs = dataloader.buildFS(train_path)
    print('Corpus size: %d' % len(fs.inodes))
    dataset = dataloader.FSDataset(fs, train_path, max_time_window=100, max_seq_size=20)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=dataloader.pad_batch_fn, shuffle=True)
    
    model = LSTMfs_(fs)
    optim = opt.Adam(model.parameters())
    
    for i, batch in enumerate(dataloader):
        optim.zero_grad()
        model.forward(batch[0])
        break