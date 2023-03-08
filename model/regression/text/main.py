import word2Sequence
import pickle
from dataset import tokenlize
import pandas as pd
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    ws = word2Sequence.word2Sequence()
    lines = np.array(pd.read_csv('../data/IMDB Dataset.csv')).tolist()
    for item in tqdm(lines):
        tokens = tokenlize(item[0])
        ws.fit(tokens)
    ws.bulid_vocab(min_count=1)
    pickle.dump(ws,open('../model/ws.pkl','wb'))
    print(len(ws))

