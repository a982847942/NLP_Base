import  torch
import numpy as np
# print(torch.__version__)
# print(np.array([[1,2,3],[4,5,6]]))
# print(torch.tensor(np.array([[1,2,3],[4,5,6]])))
# print(torch.tensor([1,2,3]))
# print(torch.tensor(1))
# print(np.empty([3,4]))

if __name__ == '__main__':
    voca_dict = {'a':0,'b':1,'d':2,'c':3}
    idx2word = {i:w for i,w in enumerate(voca_dict)}
    print(idx2word)
