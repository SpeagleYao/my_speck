from crypto.speck import speck
import numpy as np

NUM_TRAIN_PAIRS = 10**7
NUM_TEST_PAIRS = 10**5
ROUNDS = 5

speck = speck()
X, Y = speck.generate_train_data(NUM_TRAIN_PAIRS, ROUNDS)
np.save("./data/train_data_"+str(ROUNDS)+"r.npy", X)
np.save("./data/train_label_"+str(ROUNDS)+"r.npy", Y)
X, Y = speck.generate_train_data(NUM_TEST_PAIRS, ROUNDS)
np.save("./data/test_data_"+str(ROUNDS)+"r.npy", X)
np.save("./data/test_label_"+str(ROUNDS)+"r.npy", Y)