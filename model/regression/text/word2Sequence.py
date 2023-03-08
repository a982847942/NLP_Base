import numpy as np


class word2Sequence():
    UNK_TAG = 'UNK'  # unknow
    PAD_TAG = 'PAD'  # padding
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}
        self.fited = False

    def to_index(self, word):
        assert self.fited == True
        return self.dict.get(word, self.UNK)

    def to_word(self, index):
        assert self.fited
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    def __len__(self):
        return len(self.dict)

    def fit(self, sentences, ):
        for sentence in sentences:
            self.count[sentence] = self.count.get(sentence, 0) + 1
            # for item in sentence:
            #     self.count[item] = self.count.get(item, 0) + 1

    def bulid_vocab(self,min_count=1, max_count=None, max_feature=None):
        if min_count is not None:
            self.count = {key: value for key, value in self.count.items() if value >= min_count}
        if max_count is not None:
            self.count = {key: value for key, value in self.count.items() if value <= max_count}
        if isinstance(max_feature, int):
            temp = sorted(list(self.count.items()), key=lambda x: x[1], reverse=True)[max_feature]
            self.count = dict(temp)
        for w in self.count:
            self.dict[w] = len(self.dict)
        self.fited = True
        # 反向字典
        self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        '''
        把句子转换为向量
        '''
        assert self.fited
        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)
        if max_len is not None and len(sentence) > max_len:
            sentence = sentence[0:max_len]
        for index, word in enumerate(sentence):
            r[index] = self.to_index(word)
        return np.array(r, dtype=np.int64)

    def inverse_transform(self, indices):
        '''
        数组转换为句子
        '''
        sentence = []
        for i in indices:
            sentence.append(self.to_word(i))
        return sentence
    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    w2s = word2Sequence()
    w2s.fit([['我','是','谁'],['我','我','谁']])
    print(w2s.dict)
    print(w2s.transform(['我','爱','中','我']))
    print(w2s.inverse_transform([2,0,0,4]))
