import numpy as np
import os
import torch

entries = os.listdir('file/')

my_array =[]
entries.sort()
print(entries)
for i in entries:
    ckpt = torch.load(f"file/{i}",map_location=torch.device('cpu'))
    val = int(i.split('.')[0][1:])
    for keys in ckpt:
        my_array.append(ckpt[keys]['latent'].tolist())


np.save("trainY", my_array)
print(my_array)
