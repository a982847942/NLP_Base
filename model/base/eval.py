import config
import numpy as np
import torch
from MySeq2Seq import Seq2seq
data = [str(i) for i in np.random.randint(0,1e9,size=[100])]
data = sorted(data,key=lambda x:len(x),reverse=True)
input_length = torch.LongTensor([len(i) for i in data])
input = torch.LongTensor([config.num_sequence.transform(list(i),config.max_len) for i in data])

seq2seq = Seq2seq()
seq2seq.load_state_dict(torch.load(config.model_save_path))

indices = seq2seq.evaluate(input,input_length)
indices = np.array(indices).transpose()
result = []
for line in indices:
    temp_result = config.num_sequence.inverse_transform(line)
    cur_line = ""
    for word in temp_result:
        if word == config.num_sequence.EOS_TAG:
            break
        cur_line += word
    result.append(cur_line)
print(data[:10])
print(result[:10])
