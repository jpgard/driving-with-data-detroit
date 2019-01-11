import pandas as pd
import json
from tensor_utils import create_time_feat
import re
from collections import defaultdict

VEHICLES_FP = 'raw-data/vehicles.csv'
MAINTENANCE_FP = 'raw-data/maintenance.csv'

def make_vehicle_sequences(df, v, seq_col = 'System Description', method = "normal"):
    #initialize seq; data structure containing one tuple for each vehicles' job sequences
    seq = []
    outfile = "./freq-pattern-data/seqs/{}_seqs.txt".format(v)
    # filter
    df = df[df['make_model'] == v]
    for uid in df['Unit#'].unique():
        uid_df = df[df['Unit#'] == uid].sort_values('WO_open_date')
        uid_seq = uid_df[seq_col].tolist()
        seq.append(uid_seq)
    # write output to text file; one vehicle sequence per row; for spark
    if method == "tf":
        with open(outfile, 'w') as f:
            for s in seq:
                # remove any punctuation
                s_clean = [re.sub('[^0-9a-zA-Z]+', '', item) for item in s]
                s_clean.append('.') # add period to make it like ptb example data
                # write
    with open(outfile, 'w') as f:
        for s in seq:
            f.write(json.dumps(s)+'\n')
    return seq


def main(min_year = 2010, max_year = 2017):
    """
    Create a dictionary with key = make_model and value = nested list of maintenance sequences for each unique vehicle of that make/model
    :param max_year: 
    :return: 
    """
    # initialize dict to contain nested list of maintenance sequences by make/model
    maint_dict = defaultdict(list)
    # read, filter, and join data
    v = pd.read_csv(VEHICLES_FP)
    v = v[(v.Year >= min_year) & (v.Year <= max_year)]
    m = pd.read_csv(MAINTENANCE_FP)
    vm_df = pd.merge(v, m, left_on='Unit#', right_on='Unit No')
    vm_df = create_time_feat(vm_df, type = "date", col_name="WO_open_date")
    vm_df['Model'] = vm_df['Model'].apply(lambda x: re.sub('[/()`?]', '', x))
    vm_df["make_model"] = vm_df['Make'].map(lambda x: str(x) + "_") + vm_df['Model'].map(lambda x: str(x).strip().replace(' ', '_'))
    for v in vm_df.make_model.unique():
        print("Generating vehicle sequences for {}".format(v))
        s = make_vehicle_sequences(vm_df, v)
        maint_dict[v] = s
    return maint_dict

if __name__ == "__main__":
    main()