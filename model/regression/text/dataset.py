from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import torch
from load import ws,max_len,batch_size


def tokenlize(content):
    content = re.sub("<.*?>", " ", content)
    filters = ['\.', '\t', '\n', '\x97', '\x96', '#', '$', '%', '&']
    content = re.sub("|".join(filters), " ", content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


class ImdbDataSet(Dataset):
    def __init__(self):
        self.lines = np.array(pd.read_csv('../data/IMDB Dataset.csv')).tolist()
        # print(self.lines[0][1])

    def __getitem__(self, item):
        label = 1 if self.lines[item][1] == 'positive' else 0
        tokens = tokenlize(self.lines[item][0])
        return tokens, label

    def __len__(self):
        return len(self.lines)

def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    # print(batch)
    labels = torch.LongTensor(np.array(batch[1]))
    texts = batch[0]
    texts = [ws.transform(i,max_len=max_len) for i in texts]
    del batch
    texts = torch.LongTensor(np.array(texts))
    return texts, labels

def getDataloader(train=True):
    dataset = ImdbDataSet()
    # print(dataset[0][0])
    # print(dataset[0][1])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    # dataset = ImdbDataSet()
    # print(dataset[0])
    for index,(content,target) in enumerate(getDataloader()):
        print(index)
        print(content)
        print(target)
        break