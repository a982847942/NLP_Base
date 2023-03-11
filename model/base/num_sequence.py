class Num_sequence:
    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    PAD = 0
    UNK = 1
    def __init__(self):
        self.dict = {self.PAD_TAG:self.PAD,self.UNK_TAG:self.UNK}
        for i in range(10):
            self.dict[str(i)] = len(self.dict)
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len):
        if len(sentence) > max_len:
            sentence = sentence[:max_len]

        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
        result = [self.dict.get(i, self.PAD_TAG) for i in sentence]
        return result
    def inverse_transform(self,indices):
        return [self.inverse_dict.get(i,self.UNK_TAG) for i in indices]
if __name__ == '__main__':
    sequence = Num_sequence()
    print(sequence.dict)
    print(sequence.transform(['0', '5', '4', '7']))
    print(sequence.inverse_transform([2, 7, 6, 9, 1]))