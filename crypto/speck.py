import numpy as np

class speck():

    def __init__(self, alpha=7, beta=2, block_size=32, key_size=64):
        self.alpha = alpha
        self.beta = beta
        self.block_size = block_size
        self.word_size = block_size/2
        self.key_size = key_size

    # Rotation to left
    def rol(x, t):
        x << t
        return ()

    # Rotation to right
    def ror():
        pass

    # Generate keys for nr rounds
    def expand_key():
        pass

    # Enc for one round
    def enc_one_round():
        pass

    # Multi-rounds enc using enc_one_round
    # Input p would be array of plaintext_l and plaintext r, each with 16 bits
    # nr represents encryption rounds
    def encrypt(p, nr):
        pass

    # Dec for one round
    def dec_one_round():
        pass

    # Multi-rounds dec using dec_one_round
    def decrypt():
        pass

    # test whether the code is right
    def test_vector():
        pass