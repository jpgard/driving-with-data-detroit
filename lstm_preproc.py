import pandas as pd
import os
import json
import re
from sklearn.utils import shuffle

vehicles = ['DODGE_CHARGER', 'FORD_CROWN_VIC', 'CHEVROLET_IMPALA']
TRAINFILE = "/Users/joshgardner/Documents/UM-Graduate/MDST/d4gx-detroit-vehicles/ptb" \
            "-data-copy/ptb.train.txt"
TESTFILE = "/Users/joshgardner/Documents/UM-Graduate/MDST/d4gx-detroit-vehicles/ptb" \
           "-data-copy/ptb.test.txt"
VALFILE = "/Users/joshgardner/Documents/UM-Graduate/MDST/d4gx-detroit-vehicles/ptb-data" \
          "-copy/ptb.valid.txt"


def main(dir):
    data = []
    for v in vehicles:
        print("formatting data for {}".format(v))
        file = v + "_seqs.txt"
        # read data into list
        with open(os.path.join(dir, file), 'r') as f:
            for line in f.readlines():
                data.append(json.loads(line))
    data = shuffle(data)
    n_train = len(data) // 2  # 50% of data for training
    n_val = (len(
        data) - n_train) // 2  # 25% of data for validation; other 25% for testing

    # todo: random train/test/valid split; write to files
    with open(TRAINFILE, 'w') as f:
        for seq in data[:n_train]:
            seq_clean = [re.sub('[^0-9a-zA-Z]+', '', item) for item in seq]
            f.write(' '.join(seq_clean) + ' \n')
    with open(VALFILE, 'w') as f:
        for seq in data[n_train: (n_train + n_val)]:
            seq_clean = [re.sub('[^0-9a-zA-Z]+', '', item) for item in seq]
            f.write(' '.join(seq_clean) + ' \n')
    with open(TESTFILE, 'w') as f:
        for seq in data[(n_train + n_val):]:
            seq_clean = [re.sub('[^0-9a-zA-Z]+', '', item) for item in seq]
            f.write(' '.join(seq_clean) + ' \n')
    return None


if __name__ == "__main__":
    main(dir='./freq-pattern-data/seqs')
