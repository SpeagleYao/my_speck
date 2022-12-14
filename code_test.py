# I always need a code_test area to check how the code works

import numpy as np
from os import urandom
from crypto.speck import speck
from models import *

import torch.optim as opt
import torch.optim.lr_scheduler as sch

base_lr = 2e-3
max_lr = 1e-4

model = ResNet_Gohr(1)

optimizer = opt.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)
scheduler = sch.CyclicLR(optimizer, base_lr, max_lr, 9, 1, cycle_momentum=False, verbose=True)

for i in range(20):

    optimizer.step()
    scheduler.step()
    # print(optimizer.state_dict()['param_groups'][0]['lr'])

# def WORD_SIZE():
#     return 16

# MASK_VAL = 2 ** WORD_SIZE() - 1

# def rol(x, k):
#     return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))

# def rol_um(x, k):
#     return((x << k)) | (x >> (WORD_SIZE() - k))

# x = np.frombuffer(urandom(2*10),dtype=np.uint16)
# print(x)
# x = urandom(1)
# print(x)
# x = np.frombuffer(x,dtype=np.uint16)
# print(x)
# print(rol(x, 4))
# print(rol_um(x, 4))
# print((x+2**20)&MASK_VAL)

# k = np.frombuffer(urandom(8*5), dtype=np.uint16).reshape(4, -1)
# print(k.shape)
# print(k)
# print(k[3])
# k = list(reversed(k))
# print(k[0])
# print(k[1:])

# n = 10
# x = np.frombuffer(urandom(2*n),dtype=np.uint16)
# print(x.shape[0])

# speck = speck()
# X, Y = speck.generate_train_data(10, 22)
# print(X.shape)
# print(Y)