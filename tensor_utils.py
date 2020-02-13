# utility functions for tensor decomposition
from datetime import datetime
import numpy as np
import pandas as pd
import itertools
import re

# The date format used in the data files.
DATE_FORMAT = '%m/%d/%y'


def create_time_feat(maintenance_df, type="year", col_name="Year_WO_Opened"):
    # create column based on year or month/year of work order open date
    if type == "year":
        # time column for year of work order open date
        maintenance_df[col_name] = maintenance_df['WO Open Date'].apply(
            lambda x: datetime.strptime(x, DATE_FORMAT).year)
    if type == "month_year":
        # time column for (one-indexed) month/year of work order open date, starting
        # from first month/year in data
        min_year = maintenance_df['WO Open Date'].apply(
            lambda x: datetime.strptime(x, DATE_FORMAT).year).min()
        maintenance_df['month'] = maintenance_df['WO Open Date'].apply(
            lambda x: datetime.strptime(x, DATE_FORMAT).month)
        maintenance_df['year'] = maintenance_df['WO Open Date'].apply(
            lambda x: datetime.strptime(x, DATE_FORMAT).year - (min_year - 1))
        maintenance_df[col_name] = 12 * maintenance_df['year'] + maintenance_df['month']
        maintenance_df.drop(['month', 'year'], axis=1, inplace=True)
    if type == "date":
        maintenance_df[col_name] = maintenance_df['WO Open Date'].apply(
            lambda x: datetime.strptime(x, DATE_FORMAT))
    if type == "vehicle_year":
        # time column for year of work order open date relative to vehicle purchase year
        maintenance_df['year_wo_opened'] = maintenance_df['WO Open Date'].apply(
            lambda x: datetime.strptime(x, DATE_FORMAT).year)
        maintenance_df[col_name] = maintenance_df['year_wo_opened'] - maintenance_df[
            'Year']
        maintenance_df.drop(['year_wo_opened'], axis=1, inplace=True)
        # drop any vehicles which had maintenance BEFORE purchase year; this means
        # purchase year data is incorrect
        maintenance_df = maintenance_df[maintenance_df[col_name] >= 0]
    return maintenance_df


def cross_prod(df, cols):
    # return a single dataframe with cross product of values in cols; should only be
    # passed 2 columns
    assert len(cols) == 2
    to_cross = []
    for col in cols:
        col_df = pd.DataFrame(df[col].unique())
        col_df['key'] = 1
        to_cross.append(col_df)
    df_out = pd.merge(to_cross[0], to_cross[1], on='key', how='outer').drop(['key'],
                                                                            axis=1)
    df_out.columns = cols
    return df_out


def one_index(df, col):
    # make column one-indexed, using lowest value as one
    smallest_val = min(df[col])
    df[col] = df[col] - (smallest_val - 1)
    return df


def factorize_column(df, col, outfile):
    # factorizes categorical column, encoding using numeric ids instead; writes a
    # lookup table to outfile
    # create column and write lookup table to output
    df['temp'] = pd.factorize(df[col])[0] + 1
    lkp_df = df[[col, 'temp']].drop_duplicates().rename(columns={'temp': 'id'})
    lkp_df.to_csv(outfile, header=True, index=False)
    # drop original col and replace with new numeric-id col temp
    df_out = df.drop(col, axis=1).rename(columns={'temp': col})
    return df_out


def make_pre_tensor(df, lkp_suffix,
                    modes=['Unit#', 'System Description', 'year_WO_Opened']):
    # create 3-way tensor with dimensions of maintenance_df_modes
    mode_1 = modes[0]
    mode_2 = modes[1]
    mode_3 = modes[2]
    # make dummies for each category of mode_2; then create df that has an entry for
    # every (mode_1,mode_3) combo
    # so layers of tensor have same dimensions; then sub by (mode_1,mode_3)
    df = pd.concat([df[mode_1], df[mode_3], pd.get_dummies(df[mode_2])], axis=1)
    crossprod = cross_prod(df, cols=[mode_1, mode_3])
    df = pd.merge(crossprod, df, on=[mode_1, mode_3], how='left') \
        .fillna(0) \
        .groupby([mode_3, mode_1]) \
        .sum() \
        .reset_index()
    # create "long" format instead of 3-d tensor
    value_cols = [x for x in df.columns if x != mode_1 and x != mode_3]
    pre_tensor = pd.melt(df, id_vars=[mode_1, mode_3], value_vars=value_cols)
    # convert each variable to take numeric id values, starting at 1, instead of
    # original values
    # mode_1
    pre_tensor = factorize_column(pre_tensor, col=mode_1,
                                  outfile='./tensor-data/{1}/{0}_{1}_lkp.csv'.format(
                                      re.sub('[^0-9a-zA-Z]+', '', mode_1), lkp_suffix))
    # mode_2
    pre_tensor = factorize_column(pre_tensor, col='variable',
                                  outfile='./tensor-data/{1}/{0}_{1}_lkp.csv'.format(
                                      re.sub('[^0-9a-zA-Z]+', '', mode_2), lkp_suffix))
    # mode_3
    pre_tensor = one_index(pre_tensor, mode_3)
    # reorder and rename columns
    pre_tensor = pre_tensor.rename(columns={"variable": mode_2})
    cols = [mode_1, mode_2, mode_3, 'value']
    return (pre_tensor[cols])
