from torch.utils.data import Dataset, DataLoader
import torch
import config
import numpy as np


class numDataset(Dataset):
    def __init__(self):
        np.random.seed(10)
        self.data = np.random.randint(0, 1e8, size=500000)
        pass

    def __getitem__(self, item):
        # return self.data[item]
        context = list(str(self.data[item]))
        label = context + ["0"]
        context_length = len(context)
        label_length = len(label)
        return context, label, context_length, label_length

    def __len__(self):
        return len(self.data)


def collate_fn(context):
    # print(context)
    context = sorted(context, key=lambda x: x[3], reverse=True)
    context, label, context_length, label_length = list(zip(*context))
    # print(config.num_sequence.dict)
    # print('context:',context)
    # print('label:',label)
    context = torch.LongTensor([config.num_sequence.transform(i, max_len=config.max_len) for i in context])
    label = torch.LongTensor([config.num_sequence.transform(i, max_len=config.max_len + 1,add_eos=True) for i in label])
    context_length = torch.LongTensor(context_length)
    label_length = torch.LongTensor(label_length)
    return context, label, context_length, label_length


num_dataset = numDataset()
dataloader = DataLoader(dataset=num_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
if __name__ == '__main__':
    for index, (content, label, content_length, label_length) in enumerate(dataloader):
        print(index)
        print(content)
        print(label)
        print('*' * 10)
        print(content_length)
        print('*' * 10)
        print(label_length)
        break
