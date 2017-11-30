import numpy as np
import pandas as pd
from IPython.display import display, HTML
import re
from collections import Counter

# from isoweek import Week
from pandas_summary import DataFrameSummary
import operator
from matplotlib import pyplot as plt
import pickle 

import multiprocessing as mp
import datetime as dt
from tqdm import tqdm_notebook
import scipy.stats as st
import scipy

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn.model_selection import train_test_split

from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
from tsfresh.feature_extraction import MinimalFCParameters, feature_calculators, EfficientFCParameters


from subprocess import call




if __name__ == '__main__': 

    dev = pd.read_csv('data/train_1.csv')


    t_index = np.arange(dev.shape[0])
    np.random.seed(19)
    np.random.shuffle(t_index)
    # t_index[0:20]
    ix_train  = t_index[:int(0.8*len(t_index))]
    ix_valid  = t_index[int(0.8*len(t_index)):]


    train_window_end = dev.shape[1]- 60 - 10
    train = dev[['Page']+list(dev.columns[1:train_window_end])]


    #target variable for feature selection 
    nan_medians = train.apply(lambda x: x[1:].quantile(0.5) ,axis=1)
    nan_medians = nan_medians.fillna(0.).values


    train_flattened = pd.melt(train, id_vars='Page', var_name='date', value_name='Visits')
    print ("flattened train" )


    settings = EfficientFCParameters()#MinimalFCParameters()

    features_filtered = extract_relevant_features(train_flattened.dropna(), nan_medians,
    default_fc_parameters=settings,
    column_id="Page", column_sort="date")

    features_filtered.reset_index(inplace=True)
    features_filtered.to_csv("data/cache/ts_features_filtered_dev.csv",index=False)


    call(["sudo", "shutdown","now"])
                                                 