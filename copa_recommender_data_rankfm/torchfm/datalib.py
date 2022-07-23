# Generate Datasets and DataLoaders

# Library to support rankfm
# Remove modules not used once this works

import pandas as pd
import pandas_options
from torch.utils.data import Dataset
from fastcore.all import L
import torch

#import numpy as np
#import sys
#from rankfm.rankfm import RankFM
#from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall, diversity
#import function_lib as flib
#import rankfmlib as fmlib

#import torch.utils.data import DataLoader, Dataset

def cat2dict(series):
    """
    Convert back and forth between a categorical variable and a dictionary
    """
    uniques = series.unique()
    idx2a = dict(zip(range(len(uniques)), uniques))
    a2idx = dict([(v,k) for k,v in idx2a.items()])
    return idx2a, a2idx

class myDataset(Dataset):
    def __init__(self, dct, data):
        """
        Arguments
        ---------
        dct: dictionary
            Contains all parameters of the model, including data

        data:  (DataFrame)
            Data to turn into a dataset (train, valid, test). This data 
            is also accessible from the dicitonary. 
        """
        if not isinstance(data, pd.DataFrame):
            print("The `data` argument must be a DataFrame")

        self.df_ = dct['df_with_attrib']
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx])# .values
#---------------------------------------------------------------------------

# This method comes from newlib.py in rankfm/ (I should merge rankfm and torchfm, UNLESS I can duplicate results with fm)
def read_data_attributes_single_file(in_file, age_cuts=None, overwrite_cache=False, dct=None, continuous_attrib=False,
    user_attrib_cols=None, item_attrib_cols=None):
    """
    in_file (string)
        File containing all records with member and destination attributes. Note that there is 
        no column identification of which columns are which attributes (TODO IN FUTURE). 
        Extract from this file user and item attributes in DataFrame format. This is required for rankfm. 
        However, for torchfm, each row contains the attributes, so there is no need to do this. 

    age_cuts : list
        Defines the break between different member age categories.

    overwrite_cache: Bool [False]
        If True, overwrite the cache even if present

    dct: dictionary
        Dictionary with values that will be merged with new information from this method

    continuous_attrib: Bool [False]
        If True: temperature, long/lat/altitude are continuous variables

    user_attrib_cols: list of str [None]
        List of user attributes 

    item_attrib_cols: list of str [None]
        List of item attributes

    Return
    ------
    interact_dct: dictionary
        New keys are: 
            'df_members': data frame of members / destinations
            'df_user_attrib': user attributes, one-hot encoded
    interact_dct['df_members'] = df_members
    interact_dct['df_user_attrib'] = df_user_attrib
    interact_dct['df_item_attr'] = df_item_attr
    df (DataFrame)
       u Dictionary with a minimum of four columns (userID (member), itemID (Destination), rating, year))
        The year column is the year of booking

    Notes
    -----
    - On exit, the data is ordered by flight-departure time
    - The raw data read in is cached in memory (alternatively, stored in binary format) for faster retrieval. 
    - I DO NOT remove duplicate MEMBER_ID/D pairs at this stage.  This is only done for the training/validation/testsing files.
    """

    if dct:
        interact_dct = dct
    else:
        interact_dct = {}

    # The leading underscores indicates a private variable by convention
    # Create a dictionary that persists across function invocations
    if not hasattr(read_data_attributes_single_file, '_dct'):
        read_data_attributes_single_file._dct = {'raw_data' : None }
    _dct = read_data_attributes_single_file._dct

    if overwrite_cache:
        read_data_attributes_single_file_dct['raw_data'] = None


    if age_cuts == None:
        age_cuts = [0, 30, 50, 70, 120]

    # Ensure that the last value of age_cuts is >= 120 to capture all age groups
    if age_cuts[-1] < 100:
        age_cuts.append(120)

    if age_cuts[-1] < 120:
        age_cuts[-1] = 120

    # User and item attributes are across entire dataset. They are restricted to the training set at a later stage.

    # age_at_flight changes for different flights, but we ignore this over a 5-year period
    # Age at flight is not really good. Actual age would be better. But that depends on frame of reference. 
    # So perhaps young, middle, old, would be better. Eg. [0-30], [30-60], [60+]? TODO. 

    if user_attrib_cols == None:
        #user_attrib_cols = ['TRUE_ORIGIN_COUNTRY','ADDR_COUNTRY','age_departure']
        user_attrib_cols = L('age_departure','GENDER','TRUE_ORIGIN_COUNTRY')

    if item_attrib_cols == None:
        item_attrib_cols = L()

    all_attrib_cols = L("MEMBER_ID","D") + user_attrib_cols + item_attrib_cols


    attrib_file = in_file
    if isinstance(_dct['raw_data'], pd.DataFrame):
        df_ = _dct['raw_data']
    else:
        df_ = pd.read_csv(attrib_file)
        # Sort data according to flight date
        df_ = df_.sort_values("FLIGHT_DATE")
        _dct['raw_data'] = df_

    df_.rename(columns={'age_at_flight':'age_departure'}, inplace=True)

    # Only keep attribute columns
    df_ = df_[all_attrib_cols]

    if 'GENDER' in user_attrib_cols:
        df_ = pd.get_dummies(df_, prefix=['gender'], columns=['GENDER'])

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Each member/ID pair should have only a single flight (do not take temporal data into account)

    # This drop_duplicates should not be done at this point, otherwise, I will never be able to predict
    # a destination in the validation set that was present in the original set for a particular member .
    # This explains why I get worse results when I do not filter out previous flights. 
    # df_ = df_.drop_duplicates(['MEMBER_ID','D'])  # already dropped, but let us make sure. <<<< NOT HERE

    # Create categories for member ages
    if not continuous_attrib:
        df_.loc[:,'age_departure'] = pd.cut(df_.age_departure, bins=age_cuts)

    df_members = df_[['MEMBER_ID', 'D']]

    #------------------------------------------------------------
    # Read item/Destination attributes
    #df1 = pd.read_csv("temp_massaged.csv")
    df1 = pd.read_csv("temp_long_lat_height.csv")
    # Initial experimentation: with min/max avg temperatures during the year
    df1 = df1.drop(["avg_wi", "avg_sp", "avg_su", "avg_fa"], axis=1)

    # Tranform temperatures into cateogical variables
    # Try to have the same number in each category? But that might not lead to 
    # relevant categories. 

    # Attributes will be considered as scalars. Divide by 100 for normalization
    # Normalize the continuous variables to (hopefully) improve inference.
    df1['avg_yr_l'] = df1.avg_yr_l / 100.
    df1['avg_yr_h'] = df1.avg_yr_h / 100.
    df1['LON_DEC']  = df1.LON_DEC / 100.
    df1['LAT_DEC']  = df1.LAT_DEC / 100.
    df1['HEIGHT']   = df1.HEIGHT / df1.HEIGHT.max()

    if not continuous_attrib:
        yr_l_cuts = [-20,40,60,80,120]
        yr_h_cuts = [-20,40,60,80,120]
        df1['avg_yr_l'] = pd.cut(df1.avg_yr_l, bins=yr_l_cuts)
        df1['avg_yr_h'] = pd.cut(df1.avg_yr_h, bins=yr_h_cuts)

    if not continuous_attrib:
        altitude_cuts = [0,1000,2000,3000]
        long_cuts = [-1.3, -1., -.7, -.4, 0.]
        lat_cuts = [-0.30, -0.15, 0., 0.15, 0.30, 0.45]
        df1['HEIGHT']  = pd.cut(df1.HEIGHT,  bins=altitude_cuts)
        df1['LON_DEC'] = pd.cut(df1.LON_DEC, bins=long_cuts)
        df1['LAT_DEC'] = pd.cut(df1.LAT_DEC, bins=lat_cuts)

    interact_dct['age_cuts'] = str(age_cuts)

    if continuous_attrib:
        df_item_attr = df1.copy()
        df_item_attrib = df_item_attr
    else:
        interact_dct['long_cuts']     = str(long_cuts)
        interact_dct['lat_cuts']      = str(lat_cuts)
        interact_dct['yr_l_cuts']     = str(yr_l_cuts)
        interact_dct['yr_h_cuts']     = str(yr_h_cuts)
        interact_dct['altitude_cuts'] = str(altitude_cuts)

        df_item_attr = df1.copy()
        df_item_attrib = pd.get_dummies(df_item_attr, 
            prefix=['yr_l','yr_h','lon','lat','alt'], 
            columns=['avg_yr_l','avg_yr_h','LON_DEC','LAT_DEC','HEIGHT'])

    print("df_item_attrib columns: ", list(df_item_attrib.columns))
    print("df_item_attrib shape: ", df_item_attrib.shape)

    # Merge item_attributes with main data df_

    #df_ = df_.drop_duplicates() 
    df_ = df_.merge(df_item_attr, how='left', on='D')
    df_['MEMBER_ID'] = df_['MEMBER_ID'].astype(int)

    # CREATE USER ATTRIBUTES

    interact_dct['df_members']   = df_members
    interact_dct['df_with_attrib']   = df_  # Are there any nans? 
    #interact_dct['df_user_attr'] = df_user_attrib
    #interact_dct['df_item_attr'] = df_item_attr

    # A case where inplace=True is useful: the change occurs in interact_dct['df_with_attrib']
    # if inplace=False, the dictionary entry remains unchanged. 
    df_.drop(['IATA','TRUE_ORIGIN_COUNTRY'], axis=1, inplace=True) # add to attributes later (categorical)

    return interact_dct

#---------------------------------------------------------------------------
def train_valid_dct(dct, train_perc, valid_perc, temporal=False, shuffle=True):
    """
    - Split a dataframe into train, validation, testing sets. 
    - Use the dictionary 'dct' to pass information. 
    - There is no return function. 
    - The DataFrame 'df_members' is assumed to be ordered accordng to 'FLIGHT_DATE' timestamp. 
    - See function 'train_valid' for more detail. 
    """
    data_df = dct['df_with_attrib']
    print(data_df.shape)

    dftrain, dfvalid, dftest = train_valid(data_df, train_perc, valid_perc, shuffle=shuffle, temporal=temporal)
    dct['data_train'] = dftrain
    dct['data_valid'] = dfvalid
    dct['data_test']  = dftest

def train_valid(x, train_perc, valid_perc, temporal=False, shuffle=True):
    """
    Split a dataframe into train, validation, testing sets

    Parameters
    ----------
    x : pandas DataFrame, assumed ordered by flight date

    train_perc, valid_perc: float
        percent of training data [0,1]
        percent of validation data [0,1]

    shuffle:  [True]
        whether to shuffle the data or not, without replacement

    temporal: [True]
        If True, flights in the validation set take place after flights in the training set. 
        IF False, flights are randomized

    Notes:
        - The first two arguments must satisfy the constraint: (train_perc + valid_perc < 1). 
        - The temporal argument suggests I must know the departure times of all flights, and this must be passed in. 
        - This suggests the use of a characteristic timestamp. It also suggests that the division between training/validation/testing
          datesets must occur earlier in the pipeline. Perhaps I should read the data in temporally increasing order, and shuffle
          only this this method. 
        - Records with duplicate (D, MEMBER_ID) are removed from train/valid/test sets. 


    Return
    ------
    x_train, x_valid, x_test : tuple of train, valid, test dataframes

    """

    if not temporal and shuffle:
        x = x.sample(frac=1)

    nb_el = len(x)

    perc_train = train_perc
    perc_valid = valid_perc; 
    perc_test = (1.-valid_perc-train_perc); 

    n_train = int(nb_el * perc_train)
    n_valid = int(nb_el * perc_valid)
    n_test  = int(nb_el * perc_test)

    """
    print("n_train: ", n_train)
    print("n_valid: ", n_valid)
    print("n_test: ", n_test)
    """

    x_train = x.iloc[0:n_train]
    x_valid = x.iloc[n_train:n_train+n_valid]
    x_test  = x.iloc[n_train+n_valid:]

    if shuffle:
        x_train = x_train.sample(frac=1)
        x_valid = x_valid.sample(frac=1)
        x_test  = x_test.sample(frac=1)


    # Ensure that each Member/Destination pair occurs only once
    x_train = x_train.drop_duplicates(['MEMBER_ID','D'])
    x_valid = x_valid.drop_duplicates(['MEMBER_ID','D'])
    x_test  = x_test.drop_duplicates(['MEMBER_ID','D'])

    """
    print("train_valid")
    print("========> x_train: \n", x_train)
    print("========> x_valid: \n", x_valid)
    print("========> x_test: \n", x_test)
    """
    return x_train, x_valid, x_test

#-------------------------------------------------------------------------------------------