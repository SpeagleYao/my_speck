import numpy as np
from os import urandom

class speck():

    def __init__(self, alpha=7, beta=2, block_size=32, key_size=64):
        self.alpha = alpha
        self.beta = beta
        self.block_size = block_size
        self.word_size = block_size/2
        self.key_size = key_size
        self.mask = 2 ** key_size - 1

    # Left Rotation
    def rol(self, x, t):
        return ((x << t) & self.mask) | (x >> (self.word_size - t))

    # Right Rotation
    def ror(self, x, t):
        return (x >> t) | ((x << (self.word_size - t)) & self.mask)

    # urandom() returns one bytes, which is 8 bits of true random number
    # As we need uint16 here, we need two bytes for one data
    def rand_uint16(self, n):
        return np.frombuffer(urandom(2 * n), dtype = np.uint16)
    
    # Generate keys for nr rounds
    # This is a 4-word key schedules
    # Thus input k: (4, batch_size)
    def expand_key(self, k, nr):
        ks = [0 for i in range(nr)]
        k = list(reversed(k))
        ks[0] = k[0]
        l = k[1:]
        for i in range(nr-1):
            l[i%3], ks[i+1] = self.enc_one_round((l[i%3], ks[i]), i)
        return ks

    # Enc for one round
    def enc_one_round(self, p, k):
        x, y = p[0], p[1]
        x = ((self.ror(x, self.alpha) + y) & self.mask) ^ k
        y = self.rol(y, self.beta) ^ x
        return x, y

    # Multi-rounds enc using enc_one_round
    # Input p would be array of plaintext_l and plaintext r, each with 16 bits
    # For batch, p: ((batch_size), (batch_size))
    # nr represents encryption rounds
    def encrypt(self, p, nr):
        x, y = p[0], p[1]
        keys = self.rand_uint16(4 * p[0].shape[0]).reshape(4, -1)
        ks = self.expand_key(keys, nr)
        for k in ks:
            x, y = self.enc_one_round((x, y), k)
        return (x, y)

    # Dec for one round
    def dec_one_round():
        pass

    # Multi-rounds dec using dec_one_round
    def decrypt():
        pass

    # test whether the code is right
    def test_vector():
        pass