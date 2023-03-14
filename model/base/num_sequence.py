class Num_sequence:
    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {self.PAD_TAG: self.PAD, self.UNK_TAG: self.UNK, self.SOS_TAG: self.SOS, self.EOS_TAG: self.EOS}
        for i in range(10):
            self.dict[str(i)] = len(self.dict)
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len, add_eos=False):

        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        sentence_len = len(sentence)
        if add_eos:
            sentence = sentence + [self.EOS_TAG]
        if sentence_len < max_len:
            sentence = sentence + [self.PAD_TAG] * (max_len - sentence_len)
        result = [self.dict.get(i, self.PAD_TAG) for i in sentence]
        return result

    def inverse_transform(self, indices):
        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]

    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    sequence = Num_sequence()
    print(sequence.dict)
    print(sequence.transform(['0', '5', '4', '7'],max_len=4))
    print(sequence.inverse_transform([2, 7, 6, 9, 1]))
