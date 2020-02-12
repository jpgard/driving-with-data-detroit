"""
Preprocessing script to create MATLAB-ready tensors summarizing detroit vehicles data.

Usage:
$ python2 tensor_preproc.py -td month_year -n log
$ python2 tensor_preproc.py -td year -n log
$ python2 tensor_preproc.py -td vehicle_year -n log
$ python2 tensor_preproc.py -td vehicle_year -n log --max_year 2011
"""

import pandas as pd
import scipy.io
from tensor_utils import *
import string
import argparse
import math

VEHICLES_FP = 'raw-data/vehicles.csv'
MAINTENANCE_FP = 'raw-data/maintenance.csv'
CUTOFF_YR = 2010


def main(normalization=None, max_year=2017):
    # read, filter, and join data
    v = pd.read_csv(VEHICLES_FP)
    v = v[(v.Year >= CUTOFF_YR) & (v.Year <= max_year)]
    m = pd.read_csv(MAINTENANCE_FP)
    vm_df = pd.merge(v, m, left_on='Unit#', right_on='Unit No')
    time_col = '{}_wo_opened'.format(args.td)
    vm_df = create_time_feat(vm_df, type=args.td, col_name=time_col)
    # make pre-tensor and write to tsv
    pt = make_pre_tensor(vm_df, modes=['Unit#', 'System Description', time_col],
                         lkp_suffix=args.td)
    if normalization:
        if normalization == 'log':
            pt['value'] = pt['value'].apply(lambda x: math.log(x + 1))
            pt.to_csv('./tensor-data/{0}/pre_tensor_{0}_{1}_{2}.dat'.format(args.td,
                                                                            normalization,
                                                                            max_year),
                      sep='\t', header=False, index=False)
        else:
            print("Must specify a valid normalization")
            raise
    else:  # no normalization specified
        pt.to_csv('./tensor-data/{0}/pre_tensor_{0}_{1}.dat'.format(args.td, max_year),
                  sep='\t', header=False, index=False)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create summary matrix from Detroit Vehicles data for analysis '
                    'MATLAB.')
    parser.add_argument('-td',
                        metavar="Time dimension (year, month_year, or vehicle_year)",
                        required=True)
    parser.add_argument('-n', metavar="Normalization method", required=False,
                        default=None)
    parser.add_argument('--max_year', metavar="Maximum vehicle year", required=False,
                        default=2017)
    args = parser.parse_args()
    main(normalization=args.n, max_year=int(args.max_year))
