import json
import re

from ddd.df_utils import generate_vehicle_maintenance_seq_df

VEHICLES_FP = 'raw-data/vehicles.csv'
MAINTENANCE_FP = 'raw-data/maintenance.csv'
MIN_YEAR = 2010
MAX_YEAR = 2017


def write_vehicle_sequences_to_file(df, v, seq_col='System Description', method="normal"):
    """
    Write a text file containing all vehicle sequences of make_model v in df.
    :param df:
    :param v:
    :param seq_col:
    :param method:
    :return:
    """
    # initialize seq; data structure containing one tuple for each vehicles' job sequences
    seq = []
    outfile = "./freq-pattern-data/seqs/{}_seqs.txt".format(v)
    # filter
    df = df[df['make_model'] == v]
    for uid in df['Unit#'].unique():
        uid_df = df[df['Unit#'] == uid].sort_values('WO_open_date')
        uid_seq = uid_df[seq_col].tolist()
        seq.append(uid_seq)
    # write output to text file; one vehicle sequence per row
    if method == "tf":
        with open(outfile, 'w') as f:
            for s in seq:
                # remove any punctuation
                s_clean = [re.sub('[^0-9a-zA-Z]+', '', item) for item in s]
                s_clean.append('.')  # add period to make it like ptb example data
                # write
    with open(outfile, 'w') as f:
        for s in seq:
            f.write(json.dumps(s) + '\n')
    return


if __name__ == "__main__":
    generate_vehicle_maintenance_seq_df()
