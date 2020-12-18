import os

import numpy as np
import pandas as pd

def compute_minority_classes(X, y):
    # read samples into dataframe, drop rule description, transpose to achieve samples=rows, features=columns
    samples = X
    samples_concat = pd.DataFrame(samples.astype(str).values.sum(axis=1), columns=['samples'])

    # read labels into dataframe (format: one binary indicator column per label)
    labels = pd.DataFrame(data=y)
    temp = labels.replace({0:1, 1:0})
    temp = temp.rename(columns={0:1})
    labels = pd.concat([labels, temp], axis=1)

    # concatenate samples with labels
    df = pd.concat([samples_concat, labels], axis=1)
    x, _ = df.shape
    # groupby will modify the existing dataframe, we need the original later
    df_copy = df.copy()
    df_copy['SampleNo'] = np.arange(len(df_copy))

    # sum label columns grouped by sample to count positives and negatives per equiv class
    # note: samples column automatically becomes index
    # Also use the earliest (sample with the smallest index) for each class so we can use it
    # to eliminate equivalent classes that are captured by a rule
    equiv_counts = df_copy.groupby('samples').agg({'SampleNo': lambda x: x.tolist(), 0: 'sum', 1: 'sum'})
    equiv_counts = equiv_counts[(equiv_counts != 0).all(1)]
    for row in equiv_counts.iterrows():
        vector = list(x * '0')
        for idx in row[1]['SampleNo']:
            vector[idx] = '1'
        equiv_counts['SampleNo'][row[0]] = "".join(vector)

    return equiv_counts['SampleNo'], equiv_counts.drop(labels='SampleNo', axis=1)