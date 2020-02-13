# usage: $ python calc_random_test_perplexity.py
import numpy as np

TEST_FP = '/Users/joshgardner/Documents/UM-Graduate/MDST/d4gx-detroit-vehicles/ptb-data' \
          '-copy/ptb.test.txt'
TRAIN_FP = '/Users/joshgardner/Documents/UM-Graduate/MDST/d4gx-detroit-vehicles/ptb' \
           '-data-copy/ptb.train.txt'

test_obs = []
with open(TEST_FP, 'r') as f:
    for line in f.readlines():
        test_obs.append(line.split())
train_obs = []
with open(TRAIN_FP, 'r') as f:
    for line in f.readlines():
        train_obs.append(line.split())
train_corpus = [y for x in train_obs for y in x]
N_WORDS = len(set(train_corpus))  # number of potential unique unigrams in sequence
test_sequence_probs_random = [(1 / float(N_WORDS)) ** len(x) for x in test_obs]
ln_test_sequence_probs_random = [np.log(x) for x in test_sequence_probs_random]
N = len(test_obs)
avg_per_word_perplexity = np.exp(-(1 / float(N)) * sum(ln_test_sequence_probs_random))
print("AVERAGE PER WORD PERPLEXITY ON TEST DATA: {}".format(avg_per_word_perplexity))
