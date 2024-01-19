import torch
import dgl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset



def collate(samples):
    """Customer collate function"""
    # The input samples is a list of pairs (graph, label, label_2).
    graphs, labels, labels_2 = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels)).unsqueeze(1)
    labels_2 = torch.tensor(np.array(labels_2)).unsqueeze(1)
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels, labels_2


class DProQAData(Dataset):
    """
    DProQ dataset
    data_path: is the file path of the data
    mode: train, val
    If you want to use DProQA for inferece, please use inference.py for it.
    """
    def __init__(self, data_path: str, mode: str):
        data_list_df = pd.read_csv(f'{data_path}/maf2+decoy_set_{mode}.txt',
                                   sep=':',
                                   header=None)

        self.x_list = data_list_df.iloc[:, 0].tolist()
        self.y_list = data_list_df.iloc[:, 1].tolist()
        self.data = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _prepare(self):
        for i in range(len(self.x_list)):
            g, tmp = dgl.data.utils.load_graphs(self.x_list[i])

            # read label
            with open(self.y_list[i]) as f:
                tmp_content = f.readlines()[0]
                label, capri = tmp_content.split(',')
                label = torch.tensor(float(label))
                capri = torch.tensor(int(float(capri)))

            self.data.append((g[0], label, capri))
