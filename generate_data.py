from crypto.speck import speck
import numpy as np
import os

NUM_TRAIN_PAIRS = 10**7
NUM_TEST_PAIRS = 10**6
ROUNDS = 7

speck = speck()
X, Y = speck.generate_train_data(NUM_TRAIN_PAIRS, ROUNDS)
if not os.path.exists("./data/"+str(ROUNDS)+"r/"): os.makedirs("./data/"+str(ROUNDS)+"r/")
np.save("./data/"+str(ROUNDS)+"r/train_data_"+str(ROUNDS)+"r.npy", X)
np.save("./data/"+str(ROUNDS)+"r/train_label_"+str(ROUNDS)+"r.npy", Y)
X, Y = speck.generate_train_data(NUM_TEST_PAIRS, ROUNDS)
np.save("./data/"+str(ROUNDS)+"r/test_data_"+str(ROUNDS)+"r.npy", X)
np.save("./data/"+str(ROUNDS)+"r/test_label_"+str(ROUNDS)+"r.npy", Y)