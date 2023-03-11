from torch.utils.data import Dataset, DataLoader
import torch
import config
import numpy as np


class numDataset(Dataset):
    def __init__(self):
        self.data = np.random.randint(0, 1e8, size=500000)
        pass

    def __getitem__(self, item):
        # return self.data[item]
        context = list(str(self.data[item]))
        label = context + ["0"]
        return context, label

    def __len__(self):
        return len(self.data)


def collate_fn(context):
    # print(context)
    context,label = list(zip(*context))
    context = torch.LongTensor([config.num_sequence.transform(i,max_len=config.max_len) for i in context])
    label = torch.LongTensor([config.num_sequence.transform(i,max_len=config.max_len + 1) for i in label])
    return context,label


num_dataset = numDataset()
dataloader = DataLoader(dataset=num_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
if __name__ == '__main__':
    for index, (content,label) in enumerate(dataloader):
        print(index)
        print(content)
        print(label)
        break
