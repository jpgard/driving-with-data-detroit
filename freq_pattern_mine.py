# usage: python freq_pattern_mine.py

import json
import os
from pymining import seqmining
import pandas as pd
from freq_pattern_preproc import generate_vehicle_maintenance_seq_df, get_vehicles_lookup_df
from nltk import ngrams
import pandas as pd
import math
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

# set value for i_ratio when sequence never occurs for other vehicles
MAX_I_RATIO = 10000


def generate_freq_seqs(data, make_mod, min_support=30):
    """
    Generate frequent sequences from a nested list of sequences.
    :param data: List of list; each element should be a list containing an ordered sequence.
    :param make_mod: make of make/model combination; used for output file naming.
    :param min_support: minimum support of sequences to consider.
    :return: pd.DataFrame of frequent sequences, support, and normalized support (support / total # jobs in data).
    """
    num_jobs = sum([len(x) for x in data])
    # calculate frequent sequences
    freq_seqs = seqmining.freq_seq_enum(data, min_support)
    if make_mod == "TERRASTAR_HORTON":
        import ipdb;
        ipdb.set_trace()
    if min_support <= 5:
        print("No frequent patterns for {}.".format(make_mod))
        return None
    if len(freq_seqs) == 0:
        print("No frequent patterns for {} at support threshold {}".format(make_mod, min_support))
        min_support -= 5
        print("Trying {}...".format(min_support))
        generate_freq_seqs(data, make_mod, min_support)
    else:
        freq_seq_df = pd.DataFrame.from_records(list(freq_seqs))
        freq_seq_df.columns = ['Sequence', 'Support']
        freq_seq_df['Normalized_Support'] = freq_seq_df['Support'] / float(num_jobs)
        freq_seq_df.sort_values('Support', ascending=False).to_csv(
            './freq-pattern-data/freq-seqs/{}_freq_seqs.csv'.format(make_mod))
    return freq_seq_df


def output_freq_seq_files(dir='./freq-pattern-data/seqs/'):
    """
    Iterate through sequence files in directory.
    :param dir: 
    :return: 
    """
    seq_files = os.listdir(dir)
    for sf in seq_files:
        print("mining frequent sequences for {}".format(sf))
        make_mod = sf.replace("_seqs.txt", "")
        data = []
        # read data into list
        with open(os.path.join(dir, sf), 'r') as f:
            for line in f.readlines():
                data.append(json.loads(line))
        generate_freq_seqs(data, make_mod)
    return None


def i_ratio(maint_seq_df, uids, vehicle_name, min_support=180, min_seqs=5, min_length=3):
    """
    Calculate i-ratio of frequent sequences for vehicle in maint_dict.
    :param maint_seq_df: dictionary with vehicle : maintenance sequences as entries.
    :param vehicle_name: vehicle name to calculate frequent sequence i-ratios for; should match a key in maint_dict.
    :param min_support: minimum support to consider; will be iteratively lowered if insufficient sequences are found.
    :param min_seqs: minimum number of sequences to return for a given vehicle.
    :param min_length: minimum length of frequent sequences to consider.
    :return: 
    """
    output_data = []
    # make 'left' and 'right' data (this is target vehicle_name, and all vehicles EXCEPT target vehicle_name respectively)
    left_data = maint_seq_df[maint_seq_df["unit"].isin(uids)]["maint_seq"]
    right_data = maint_seq_df[~maint_seq_df["unit"].isin(uids)]["maint_seq"]
    left_freq_seqs = []
    while len(left_freq_seqs) < min_seqs:
        print("Searching for frequent sequences for {} with min support {}".format(vehicle_name, min_support))
        left_freq_seqs = seqmining.freq_seq_enum(left_data, min_support)
        left_freq_seqs = [x for x in left_freq_seqs if len(x[0]) >= min_length]
        min_support -= 1
    for left_seq in left_freq_seqs:
        seq = left_seq[0]
        left_seq_support = left_seq[1]
        left_ngrams = [x for item in left_data for x in ngrams(item, len(seq))]
        left_seq_norm_support = left_seq_support / float(len(left_ngrams))
        # get support for seq in right data
        right_ngrams = [x for item in right_data for x in ngrams(item, len(seq))]
        right_matches = [x for x in right_ngrams if x == seq]
        right_seq_support = len(right_matches)
        right_seq_norm_support = right_seq_support / float(len(right_ngrams))
        try:
            i_ratio = left_seq_norm_support / right_seq_norm_support
        except ZeroDivisionError:  # left_seq never occurs in right_data
            i_ratio = MAX_I_RATIO
        # t test for difference between two population means for left_seq_norm_support and right_seq_norm_support
        counts = np.array([left_seq_support, right_seq_support])
        nobs = np.array([len(left_ngrams), len(right_ngrams)])
        z_stat, p_z = proportions_ztest(counts, nobs, value=0.05)
        seq_data = (vehicle_name, seq, left_seq_support, round(left_seq_norm_support, 4), right_seq_support,
                    round(right_seq_norm_support, 4), round(i_ratio, 2), round(z_stat, 1), round(p_z, 4))
        output_data.append(seq_data)
    return output_data


def create_i_ratio_df(maint_seq_df, uids, vehicle_name):
    results_df = pd.DataFrame(i_ratio(maint_seq_df, uids, vehicle_name))
    results_df.columns = ['Vehicle', 'Sequence', 'Left Support', 'Left Norm Support', 'Right Support',
                          'Right Norm Support', 'i-Ratio', 'z', 'P(z)']
    results_df.sort_values(by=['Left Support', 'P(z)', 'Vehicle'], ascending=[False, True, True], inplace=True)
    return results_df


def frequentist_dsm_by_makemodel(vehicles=('HUSTLER_X-ONE', 'SMEAL_SST_PUMPER', 'DODGE_CHARGER', 'FORD_CROWN_VIC'),
                                 outpath='./freq-pattern-data/i_ratios.csv'):
    # note: issue calculating frequent sequences for FREIGHTLIN_M2112V; possibly due to too few makes and identical maintenance of all vehicles
    maint_seq_df = generate_vehicle_maintenance_seq_df()
    vehicles_lookup_df = get_vehicles_lookup_df()
    i_ratio_df_list = list()
    for v in vehicles:
        uids = vehicles_lookup_df[vehicles_lookup_df["make_model"] == v]["Unit#"].tolist()
        i_ratio_df_list.append(create_i_ratio_df(maint_seq_df, uids, v))
    i_ratio_df = pd.concat(i_ratio_df_list)
    i_ratio_df.to_csv(outpath, header=True, index=False)
    print("Output written to {}".format(outpath))


def compute_cluster_membership(A, thresh=0.99):
    """
    Using BGMM, compute two clusters (in-group = 1, out-group = 1) based on each individual column of A.
    :param A: PARAFAC factor loading matrix A.
    :return: np.array with binary values, which can be interpreted as either cluster labels or in-group membership indicators.
    """
    n, R = A.shape
    cluster_membership = np.full((n, R), np.nan)
    for r in range(R):
        bgmm = BayesianGaussianMixture(n_components=2)
        loading_vector = np.reshape(A.iloc[:, r].values, (-1, 1))
        bgmm.fit(loading_vector)
        probs = bgmm.predict_proba(loading_vector)[:,0] # probability of first class
        if sum(probs > thresh) > n // 2:  # case: cluster zero is majority group; return cluster one as label
            labels = probs < thresh
        else: # case: cluster zero is minority group
            labels = probs > thresh
        cluster_membership[:, r] = labels.astype(int)
    return cluster_membership


def bayesian_dsm_from_parafac(A_matrix_fp="./tensor-data/month_year/A_monthyear_log.txt", ):
    A = pd.read_csv(A_matrix_fp, header=None)
    # makemodel =
    cluster_membership_matrix = compute_cluster_membership(A)
    np.savetxt("./tensor-data/month_year/bgmm_cluster_membership.txt", cluster_membership_matrix)
    import ipdb;ipdb.set_trace()
    # todo: write to CSV; validate these with some plotting in R; then compute frequent sequences via BDSM


if __name__ == "__main__":
    # bayesian_dsm_from_parafac()
    frequentist_dsm_by_makemodel()
