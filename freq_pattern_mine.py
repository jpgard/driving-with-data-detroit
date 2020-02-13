# usage: python freq_pattern_mine.py

import os
from pymining import seqmining
from freq_pattern_preproc import generate_vehicle_maintenance_seq_df, get_vehicles_lookup_df, \
    get_system_description_lookup_df
from nltk import ngrams
import pandas as pd
import math
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import matplotlib
from collections import Counter

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import time

# set value for i_ratio when sequence never occurs for other vehicles
MAX_I_RATIO = 10000


def bayesian_prop_test(successes, n, rope=0.01, plot=False):
    """
    Computes the posterior probability of a difference in means between two groups.
    :param successes: iterable of success counts for each group.
    :param n: iterable of total number of trials for each group.
    :param rope: region of practical equivalence.
    :return: posterior probability that the means of the two groups are not the same; that is: P(mu_1 != mu_2 | data)
    """
    ## for testing:
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1,
                        shape=len(n))  # see https://docs.pymc.io/notebooks/GLM-hierarchical-binominal-model.html
        p = pm.Binomial("successes", p=theta, observed=successes, n=n)
        trace = pm.sample(2000, tune=100,  # note: these converge relatively quickly; not much sampling needed
                          cores=1)  # note: must set cores=1 because using Python2.7; see https://github.com/pymc-devs/pymc3/issues/3255
    if plot:
        pm.plots.plot_posterior(trace)
        pm.plots.forestplot(trace)
        pm.traceplot(trace)
    # compute posterior statistics
    theta_posterior_samps = trace["theta"]
    posterior_mean = theta_posterior_samps.mean(axis=0)
    diff_in_posterior_means = abs(posterior_mean[1] - posterior_mean[0])
    trace_theta_diff_in_rope = np.apply_along_axis(lambda x: abs(x[0] - x[1]) <= rope, 1, theta_posterior_samps)
    p_rope = sum(trace_theta_diff_in_rope) / float(len(theta_posterior_samps))
    return diff_in_posterior_means, p_rope


def freq_seq_search(seqs, systems, search_min_support=180, search_max_seqs=500, search_min_seqs=5, search_min_length=2,
                    add_all_system_ngrams=False, ngram_min_support=5, ngram_range=(2, 3)):
    """
    Search for frequent sequences in seqs using a frequent sequence mining algorithm.
    :param seqs:
    :param systems:
    :param search_min_support: minimum support to consider; will be iteratively lowered if insufficient sequences are found (best to start with very large value; overestimating induces negligible additional computation).
    :param search_min_seqs: minimum number of sequences to return for a given evaluation.
    :param ngram_range: minimum,maximum length of frequent sequences to consider.
    :ngram_min_support: minimum support for ngrams from system-level search
    :search_max_seqs: maximum number of sequences to return from pattern mining search
    :return:
    """
    print("[INFO] generating frequent sequences")
    start_time = time.time()
    freq_seqs = list()
    all_systems_in_freq_seqs = False  # indicator for whether at least one frequent subsequence has been found with every system in systems
    while len(freq_seqs) < search_max_seqs:
        freq_seqs = seqmining.freq_seq_enum(seqs, search_min_support)  # set of (sequence, frequency) tuples
        freq_seqs = [x for x in freq_seqs if len(x[0]) >= search_min_length and not set(x[0]).isdisjoint(systems)]
        search_min_support -= 1
        if search_min_support == 0:
            print("[WARNING] Failed to find frequent sequences for {}".format(identifier))
            return None
        # check if all systems have been identified in at least one subsequence
        systems_in_freq_seqs = set([system for item in freq_seqs for system in item[0]])
        all_systems_in_freq_seqs = set(systems).issubset(systems_in_freq_seqs)
    if add_all_system_ngrams:
        # if not all systems, explicitly find common subsequences containing those systems
        for ngram_length in range(ngram_range[0], ngram_range[1] + 1):
            print(
            "[INFO] getting all system ngrams of length {} with min_support {}".format(ngram_length, ngram_min_support))
            ngram_obs = [ngrams(s, ngram_length) for s in seqs.tolist()]
            # "unroll" the ngram iterators, keeping only sequences which contain one of systems
            system_ngram_counts = Counter(
                [b for vehicle_bigrams in ngram_obs for b in vehicle_bigrams if not set(b).isdisjoint(systems)])
            for system in systems:
                system_ngrams = sorted(
                    [x for x in system_ngram_counts.items() if system in x[0] and x[1] >= ngram_min_support],
                    key=lambda x: x[1], reverse=True)
                freq_seqs += system_ngrams
    end_time = time.time()
    print("[INFO] found frequent sequences in {} sec".format(round(end_time - start_time)))
    return freq_seqs


def i_ratio(left_seqs, right_seqs, systems, identifier, stat_test_method):
    """
    Calculate i-ratio of frequent sequences for vehicle in maint_dict.
    :param maint_seq_df: dictionary with vehicle : maintenance sequences as entries.
    :param uids: unique vehicle ids to evaluate
    :param systems: systems to evaluate
    :param identifier: vehicle name or other identifier for this specific testing iteration.
    :param stat_test_method: technique to use for statistical inference; either "frequentist" or "bayesian".
    :return:
    """
    output_data = []
    left_freq_seqs = freq_seq_search(left_seqs, systems)
    print("[INFO] Identified {} frequent sequences for {}; conducting testing".format(len(left_freq_seqs), identifier))
    for left_seq in left_freq_seqs:
        seq = left_seq[0]
        left_seq_support = left_seq[1]
        left_ngrams = [x for item in left_seqs for x in ngrams(item, len(seq))]
        left_seq_norm_support = left_seq_support / float(len(left_ngrams))
        # get support for seq in right data
        right_ngrams = [x for item in right_seqs for x in ngrams(item, len(seq))]
        right_matches = [x for x in right_ngrams if x == seq]
        right_seq_support = len(right_matches)
        right_seq_norm_support = right_seq_support / float(len(right_ngrams))
        try:
            i_ratio = left_seq_norm_support / right_seq_norm_support
        except ZeroDivisionError:  # left_seq never occurs in right_seqs
            i_ratio = MAX_I_RATIO
        counts = np.array([left_seq_support, right_seq_support])
        nobs = np.array([len(left_ngrams), len(right_ngrams)])
        if stat_test_method == "frequentist":
            # for frequentist stat_test_method, test_statistic_value is a z-score; p is its p-value
            # t test for difference between two population means for left_seq_norm_support and right_seq_norm_support
            test_statistic_value, p = proportions_ztest(counts, nobs, value=0.05)
        elif stat_test_method == "bayesian":
            # for bayesian stat_test_method, test_statistic_value is the magnitude of the difference in posterior means theta_0 - theta_1;
            # p is the posterior probability that these means differ by more than ROPE
            test_statistic_value, p = bayesian_prop_test(successes=counts, n=nobs)
        else:
            raise NotImplementedError("stat_test_method must be either 'frequentist' or 'bayesian'.")
        seq_data = (identifier, seq, left_seq_support, round(left_seq_norm_support, 4), right_seq_support,
                    round(right_seq_norm_support, 4), round(i_ratio, 2), round(test_statistic_value, 1), round(p, 4))
        output_data.append(seq_data)
    return output_data


def create_i_ratio_df(left_seqs, right_seqs, systems, identifier, stat_test_method):
    """

    :param left_seqs:
    :param right_seqs:
    :param systems: systems in the group to evaluate.
    :param identifier: unique identifier to describe this iteration of i_ratio testing.
    :param stat_test_method: either "frequentist" or "bayesian".
    :return:
    """
    assert stat_test_method in ("frequentist", "bayesian")
    i_ratio_results = i_ratio(left_seqs, right_seqs, systems, identifier, stat_test_method=stat_test_method)
    if i_ratio_results:  # i_ratio() returns no results if frequent sequences cannot be found
        results_df = pd.DataFrame(i_ratio_results)
        results_df.columns = ['Identifier', 'Sequence', 'Left Support', 'Left Norm Support', 'Right Support',
                              'Right Norm Support', 'i-Ratio', 'z', 'P(z)']
        results_df.sort_values(by=['Identifier', 'P(z)', 'Left Support'], ascending=[False, True, True], inplace=True)
        return results_df


def compute_cluster_membership(A):
    """
    Using BGMM, compute two clusters (in-group = 1, out-group = 1) based on each individual column of A.
    :param A: PARAFAC factor loading matrix A.
    :return: np.array with binary values, with 1 indicating in-group membership and 0 otherwise.
    """
    n, R = A.shape
    cluster_membership = np.full((n, R), np.nan)
    for r in range(R):
        bgmm = BayesianGaussianMixture(n_components=2)
        loading_vector = np.reshape(A.iloc[:, r].values, (-1, 1))
        labels = bgmm.fit_predict(loading_vector)
        in_group_label = np.argmax(bgmm.means_)
        is_ingroup = labels == in_group_label
        # set the "in group" label to the group with higher posterior mean value
        cluster_membership[:, r] = is_ingroup.astype(int)
    return cluster_membership


def run_prism(A_matrix_fp, vehicle_ingroup_matrix_fp, B_matrix_fp, system_ingroup_matrix_fp,
              C_matrix_fp, time_ingroup_matrix_fp, time_colname,
              vehicle_lkp_fp, rgrid=None):
    """
    Conduct PaRafac-Informed Sequence Mining (PRISM) using the results of PARAFAC.
    :param A_matrix_fp: path to read A matrix for PARAFAC.
    :param vehicle_ingroup_matrix_fp: path to write vehicle in-group matrices.
    :param B_matrix_fp: path to read B matrix fmor PARAFAC.
    :param system_ingroup_matrix_fp: path to write system in-group matrices.
    :param C_matrix_fp: path to read C matrix fmor PARAFAC.
    :param time_ingroup_matrix_fp: path to write time in-group matrices.
    :param vehicle_lkp_fp: path to vehicle lookup df; used to ensure ordering matches A matrix precisely.
    :param rgrid: grid of values to evaluate for r; by default all factors in range(R) will be evaluated.
    :return:
    """
    A = pd.read_csv(A_matrix_fp, header=None)
    B = pd.read_csv(B_matrix_fp, header=None)
    C = pd.read_csv(C_matrix_fp, header=None)
    n, R = A.shape
    if not rgrid:
        rgrid = [x for x in range(R)]
    vehicles_lookup_df = get_vehicles_lookup_df(vehicle_lkp_fp=vehicle_lkp_fp)
    system_lookup_df = get_system_description_lookup_df()
    vehicle_ingroup_matrix = compute_cluster_membership(A)
    system_ingroup_matrix = compute_cluster_membership(B)
    time_ingroup_matrix = compute_cluster_membership(C)
    maint_seq_df = generate_vehicle_maintenance_seq_df()
    print("[INFO] writing vehicle cluster membership matrix to {}".format(vehicle_ingroup_matrix_fp))
    np.savetxt(vehicle_ingroup_matrix_fp, vehicle_ingroup_matrix)
    print("[INFO] writing system cluster membership matrix to {}".format(system_ingroup_matrix_fp))
    np.savetxt(system_ingroup_matrix_fp, system_ingroup_matrix)
    print("[INFO] writing time cluster membership matrix to {}".format(time_ingroup_matrix_fp))
    np.savetxt(time_ingroup_matrix_fp, time_ingroup_matrix)
    for r in rgrid:
        import ipdb;ipdb.set_trace()
        in_group_uids = vehicles_lookup_df.iloc[vehicle_ingroup_matrix[:, r] == 1, :]["Unit#"].tolist()
        in_group_systems = system_lookup_df.iloc[system_ingroup_matrix[:, r] == 1, :]["variable"].tolist()
        in_group_times = np.argwhere(time_ingroup_matrix[:, r] == 1).flatten().tolist()
        left_maint_seq_df = generate_vehicle_maintenance_seq_df(write_to_file=False, filter_col=time_colname,
                                                                filter_values=in_group_times)
        left_seqs = left_maint_seq_df[left_maint_seq_df["unit"].isin(in_group_uids)]["maint_seq"]
        right_seqs = maint_seq_df[~maint_seq_df["unit"].isin(in_group_uids)]["maint_seq"]
        identifier = "PARAFAC_{}".format(r)
        i_ratio_df = create_i_ratio_df(left_seqs, right_seqs, in_group_systems, identifier, stat_test_method="bayesian")
        if i_ratio_df is not None:
            outpath = './freq-pattern-data/i_ratios_{}_PARAFAC_r{}.csv'.format(time_colname, r)
            i_ratio_df.to_csv(outpath, header=True, index=False)
    return


def concatenate_maintenance_sequences(seqs, start_token="<START>", end_token="<END>"):
    """
    Concatenate all subsequences in seqs into a single sequence, with start_token and end_token indicating the beginning and ending of subsequences.
    :param seqs: iterable of sequences.
    :param start_token: token to use for subequence start.
    :param end_token: token to use for sobsequence end.
    :return:
    """
    out_seq = []
    for subseq in seqs:
        out_seq += [start_token] + ["<{}>".format(item) for item in subseq] + [end_token]
    return out_seq


if __name__ == "__main__":
    # # with month-year analysis
    run_prism(A_matrix_fp="./tensor-data/vehicle_year/A_vehicle_year_log.txt",
              vehicle_ingroup_matrix_fp="./tensor-data/vehicle_year/vehicle_ingroup.txt",
              B_matrix_fp="./tensor-data/vehicle_year/B_vehicle_year_log.txt",
              system_ingroup_matrix_fp="./tensor-data/vehicle_year/system_ingroup.txt",
              C_matrix_fp="./tensor-data/vehicle_year/C_vehicle_year_log.txt",
              time_ingroup_matrix_fp="./tensor-data/vehicle_year/monthyear_ingroup.txt",
              time_colname="vehicle_year",
              vehicle_lkp_fp="./tensor-data/vehicle_year/Unit_vehicle_year_lkp.csv",
              rgrid=(2, 14, 15))
    run_prism(A_matrix_fp="./tensor-data/month_year/A_monthyear_log.txt",
              vehicle_ingroup_matrix_fp="./tensor-data/month_year/vehicle_ingroup.txt",
              B_matrix_fp="./tensor-data/month_year/B_monthyear_log.txt",
              system_ingroup_matrix_fp="./tensor-data/month_year/system_ingroup.txt",
              C_matrix_fp="./tensor-data/month_year/C_monthyear_log.txt",
              time_ingroup_matrix_fp="./tensor-data/month_year/monthyear_ingroup.txt",
              time_colname="month_year",
              vehicle_lkp_fp="./tensor-data/month_year/Unit_month_year_lkp.csv",
              rgrid=(0, 9, 16)
              )
