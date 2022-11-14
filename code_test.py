# I always need a code_test area to check how the code works

import numpy as np
from os import urandom

def WORD_SIZE():
    return 16

MASK_VAL = 2 ** WORD_SIZE() - 1

def rol(x, k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))

def rol_um(x, k):
    return(((x << k)) | (x >> (WORD_SIZE() - k)))

x = np.frombuffer(urandom(2*10),dtype=np.uint16)

print(x)
print(rol(x, 4))
print(rol_um(x, 4))