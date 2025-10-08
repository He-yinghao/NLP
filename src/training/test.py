import torch
with open('../models/data.txt', 'r', encoding='utf-8') as f:
    data = f.read()
    print(data)
print(torch.__version__)
