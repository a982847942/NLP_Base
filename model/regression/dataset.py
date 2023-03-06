from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchtext.datasets import IMDB



data_path = r'H:\pythonProject\data\SMSSpamCollection'


class MyDataSet(Dataset):
    def __init__(self):
        self.lines = open(data_path, 'r', encoding='mac_roman').readlines()

    def __getitem__(self, item):
        curLine = self.lines[item].strip()
        label = curLine[:4].strip()
        content = curLine[4:]
        return label, content

    def __len__(self):
        return len(self.lines)


myDataset = MyDataSet()
dataLoader = DataLoader(myDataset,batch_size=2,shuffle=True,num_workers=2,drop_last=True)
if __name__ == '__main__':
    for index,(label,content) in enumerate(dataLoader):
        print(index,label,content)
        break
