import os
import glob
import random

import torch

# korean license plate characters dictionary
char_dict = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '가': 10,
}

# convert string to label (ex. '가1234' -> [10, 0, 1, 2, 3, 4]) type: torch.Tensor(int32)
def str2label_tensor(str):
    label = []
    for char in str:
        label.append(char_dict[char])
    return torch.tensor(label, dtype=torch.int32)


# convert label(torch.Tensor) to string (ex. [10, 0, 1, 2, 3, 4] -> '가1234') type: str
def label_tensor2str(label):
    str = ''
    for i in label:
        for key, value in char_dict.items():
            if i == value:
                str += key
    return str


if __name__ == '__main__':
    ret = str2label_tensor('123가1234')
    print(ret)
    ret = label_tensor2str(ret)
    print(ret)
