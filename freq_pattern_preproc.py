import pandas as pd
import json
from tensor_utils import create_time_feat
import re

VEHICLES_FP = 'raw-data/vehicles.csv'
MAINTENANCE_FP = 'raw-data/maintenance.csv'
MIN_YEAR = 2010
MAX_YEAR = 2017

def get_vehicles_lookup_df(vehicles_fp = VEHICLES_FP, min_year=MIN_YEAR, max_year=MAX_YEAR):
    v = pd.read_csv(vehicles_fp)
    v = v[(v.Year >= min_year) & (v.Year <= max_year)]
    v["make_model"] = v['Make'].map(lambda x: str(x) + "_") + v['Model'].map(
        lambda x: str(x).strip().replace(' ', '_'))
    return v


def get_system_description_lookup_df(system_description_fp="./tensor-data/vehicle_year/SystemDescription_vehicle_year_lkp.csv"):
    return pd.read_csv(system_description_fp)


def get_maintenance_lookup_df(maintenance_fp=MAINTENANCE_FP):
    return pd.read_csv(maintenance_fp)


def get_vehicle_maintenance_lookup_df(vehicles_fp=VEHICLES_FP, maintenance_fp=MAINTENANCE_FP, min_year=MIN_YEAR, max_year=MAX_YEAR):
    v = get_vehicles_lookup_df(vehicles_fp, min_year=min_year, max_year=max_year)
    m = get_maintenance_lookup_df(maintenance_fp)
    vm_df = pd.merge(v, m, left_on='Unit#', right_on='Unit No')
    vm_df = create_time_feat(vm_df, type="date", col_name="WO_open_date")
    vm_df = create_time_feat(vm_df, type="month_year", col_name="month_year")
    vm_df["month_year"] = vm_df["month_year"] - vm_df["month_year"].min() # shift down to zero
    vm_df['Model'] = vm_df['Model'].apply(lambda x: re.sub('[/()`?]', '', x))
    vm_df["make_model"] = vm_df['Make'].map(lambda x: str(x) + "_") + vm_df['Model'].map(
        lambda x: str(x).strip().replace(' ', '_'))
    return vm_df


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


def generate_vehicle_maintenance_seq_df(seq_col='System Description', write_to_file=True, filter_col=None, filter_values=None):
    """
    Create a dictionary with key = make_model and value = nested list of maintenance sequences for each unique vehicle of that make/model
    :param seq_col: column to generate sequences of
    :param write_to_file: indicator for whether sequences should be written to a text file
    :param filter_col: column name to use for filtering; with filter_values
    :param filter_values: values used to filter filter_col; only entries with one of filter_values will be kept.
    :return: 
    """
    maint_seqs = list() # tuples of (uid, makemodel, list_of_maintenance_seqs)
    # read, filter, and join data
    vm_df = get_vehicle_maintenance_lookup_df()
    print("[INFO] generating vehicle sequences")
    for vehicle_make_model in vm_df['make_model'].unique():
        make_model_uids = vm_df[vm_df['make_model'] == vehicle_make_model]['Unit#'].unique()
        if write_to_file:
            write_vehicle_sequences_to_file(vm_df, vehicle_make_model)
        for unit in make_model_uids:
            if not filter_col:
                unit_seq = vm_df[(vm_df['make_model'] == vehicle_make_model) & (vm_df["Unit#"] == unit)].sort_values('WO_open_date')[seq_col].tolist()
            else:
                assert filter_values, "must specify filter_values if using filter_col"
                unit_seq = vm_df[(vm_df['make_model'] == vehicle_make_model) & (vm_df["Unit#"] == unit) & (vm_df[filter_col].isin(filter_values))].sort_values(
                    'WO_open_date')[seq_col].tolist()
            maint_seqs.append((unit, vehicle_make_model, unit_seq))
    maint_seq_df = pd.DataFrame(maint_seqs)
    maint_seq_df.columns = ["unit", "make_model", "maint_seq"]
    return maint_seq_df


if __name__ == "__main__":
    generate_vehicle_maintenance_seq_df()
