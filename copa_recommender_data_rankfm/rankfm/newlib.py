# Library to support rankfm

import pandas as pd
import numpy as np
import sys
import pandas_options

from rankfm.rankfm import RankFM
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall, diversity
import function_lib as flib
import rankfmlib as fmlib

#----------------------------------------------------------------------------------------
def read_data_attributes_single_file(in_file, age_cuts=None): #, year_train, year_valid)
    """
    in_file (string)
        File containing all records with member and destination attributes. Note that there is 
        no column identification of which columns are which attributes (TODO IN FUTURE)

    age_cuts : list
        Defines the break between different member age categories.

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
    """

    #df = {}
    #df_members = {}
    #df_user_attrib = {}

    interact_dct = {}

    if age_cuts == None:
        age_cuts = [0, 30, 50, 70, 120]

    assert age_cuts[-1] > 120, "maximum age must be great or equal than 120"

    # age_at_flight changes for different flights, but we ignore this over a 5-year period
    cols_user_attrib = ['MEMBER_ID','TRUE_ORIGIN_COUNTRY','ADDR_COUNTRY','age_departure']
    # This should be an argument to the function
    cols_user_attrib = ['MEMBER_ID','age_departure']

    attrib_file = in_file
    df_ = pd.read_csv(attrib_file)
    print(df_.columns, df_.shape)
    # Each member/ID pair should have only a single flight (no not take temporal data into account)
    df_ = df_.drop_duplicates(['MEMBER_ID','D'])  # already dropped, but let us make sure.

    # Bin passenger ages
    df_['age_departure'] = pd.cut(df_.age_at_flight, bins=age_cuts)
    df_members = df_[['MEMBER_ID', 'D']]

    df_user_attrib = df_[cols_user_attrib].drop_duplicates('MEMBER_ID')

    # One-Hot encode the categorical attributes
    df_user_attrib = pd.get_dummies(df_user_attrib, prefix=['age_dep'], columns=['age_departure'])

    # Age at flight is not really good. Actual age would be better. But that depends on frame of reference. 
    # So perhaps young, middle, old, would be better. Eg. [0-30], [30-60], [60+]? TODO. 
    member_attr_cols = ['MEMBER_ID', 'GENDER', 'age_at_flight', 'NATIONALITY', 'ADDR_COUNTRY'] 
    member_categ_attr_df = df_[member_attr_cols]

    # Read item/Destination attributes
    df1 = pd.read_csv("temp_massaged.csv")
    # Initial experimentation: with min/max avg temperatures during the year
    df1 = df1.drop(["avg_wi", "avg_sp", "avg_su", "avg_fa"], axis=1)
    # Attributes will be considered as scalars. Divide by 100 for normalization
    df1['avg_yr_l'] = df1.avg_yr_l / 100.
    df1['avg_yr_h'] = df1.avg_yr_h / 100.
    df_item_attr = df1.copy()
    # TODO: Work with temperatures

    #print("return from read_data_attributes_single_file")
    #return df_members, df_user_attrib, df_item_attr
    interact_dct['df_members'] = df_members
    interact_dct['df_user_attr'] = df_user_attrib
    interact_dct['df_item_attr'] = df_item_attr
    return interact_dct

#-------------------------------------------------------------------------------------
def calculate_dct_with_attributes(model, nb_samples=10000, with_attrib=False, verbose=False, nb_epochs=30, loss='warp'):
    """
    Self-contained function: reads the data and calculates a hit rate
    
    Parameters:
    ----------
    nb_samples: [10000], int
        number of unique members in the full dataset (train/valid/test). If None, keep all members.

    with_attrib: [False], Bool
        Whether or not to run RankFM with attributes

    verbose: [False], Bool
        If True, print more data (not used)

    nb_epochs: [30]

    loss: ['warp']

    Return:
    -------
    interactions_dct: dictionary
        Contains useful data, keyed for easy retrieval
    """

    # Read data with attributes
    in_file = "activity_reduced_with_attributes.csv"
    member_dest_df, df_user_attrib, df_item_attrib = read_data_attributes_single_file(in_file)
    df_item_attrib = df_item_attrib.reset_index(drop=True)

    # Any shuffling and filteirng operation on member_dest_df implies that df_user_attrib must be filtered
    # to make sure it does not contain members possibly deleted from member_dest_df

    # Do not shuffle
    interactions = member_dest_df.copy()

    # Subsample data for initial testing

    all_users = interactions.MEMBER_ID.unique()
    max_nb_users = len(all_users)

    if nb_samples == None or nb_samples > max_nb_users:
        nb_samples = max_nb_users # number beyond the maximum 
    keep_nb_users = nb_samples

    # shuffle the users
    np.random.seed(1492)
    s_users = np.random.choice(all_users, size=keep_nb_users, replace=False)

    # Filter users in df_item_onebot. Only keep users with ids in s_users
    df_user_attrib = df_user_attrib[df_user_attrib['MEMBER_ID'].isin(s_users)]

    # Filter users in member_dest_df
    member_dest_df = member_dest_df[member_dest_df['MEMBER_ID'].isin(s_users)]
    assert member_dest_df.groupby(['MEMBER_ID', 'D']).size().max() == 1, "max flights by one passenger per year should be 1"

    print("member_dest_df: ", type(member_dest_df)) # dataframe
    data_train, data_valid, data_test = train_valid(member_dest_df, train_perc=0.8, valid_perc=0.1)
    print(data_train.shape, data_valid.shape, data_test.shape)
    print("data_train: ", data_train)

    # I can weight the samples in various ways. I am not doing so. 
    sample_weight = np.ones(member_dest_df.shape[0])

    interactions_dct = {
            "train": member_dest_df, 
            #"train": member_dest_df, 
            "sample_weight": sample_weight,
            "user_attrib": df_user_attrib,
            "item_attrib": df_item_attrib,
    }

    # won't work because elements of interactions_dct changed
    #fmlib.print_stats(interactions_dct)

    interactions_train = interactions_dct["train"]
    user_features = interactions_dct["user_attrib"]
    item_features = interactions_dct["item_attrib"]
    sample_weight_train = interactions_dct["sample_weight"]

    if loss != 'warp' and loss != 'bpr':
        raise("loss must be 'warp' or 'bpr'")

    #model = RankFM(factors=50, loss=loss, max_samples=50, alpha=0.01, learning_rate=0.1, learning_schedule='invscaling')

    # Drop Destinations from item_features not contained in interactions_train
    dest_feat = set(interactions_train['D'])
    item_feat = set(item_features['D'])
    items_to_remove = dest_feat - item_feat
    item_features = item_features[item_features['D'].isin(dest_feat)].reset_index(drop=True)

    # Remove duplicates from item_features (item 25  occurs twice. WHY?)
    item_features = item_features.drop_duplicates('D')

    # UP TO THIS POINT, it is always the same 

    #---------------------------------------------------------------------------------
    # Fit model

    if with_attrib:
        # No item features
        model.fit(interactions_train, user_features=user_features, sample_weight=sample_weight_train, epochs=nb_epochs, verbose=verbose)
        #model.fit(interactions_train, user_features=user_features, item_features=item_features, sample_weight=sample_weight_train, epochs=nb_epochs, verbose=verbose)
    else:
        model.fit(interactions_train, sample_weight=sample_weight_train, epochs=nb_epochs, verbose=verbose)

    # Generate model scores
    #interactions_valid = interactions_dct["valid"]
    #scores = model.predict(interactions_valid, cold_start='nan') # 'nan' or 'drop'`

    # Does not change model
    #valid_recs = model.recommend(valid_users, n_items=10, filter_previous=False, cold_start='nan')
    #valid_recs = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='nan')

    #interactions_dct['model'] = model

    return interactions_dct
#----------------------------------------------------------------------------------------
def topn_recommendations_with_attributes(model, interactions_dct, user_attributes=None, topN=3):
    """
    Generate top-N recommendations. 

    Parameters:
    ----------
    user_attributes=None, not used
    """

    interactions_train = interactions_dct["train"]
    interactions_valid = interactions_dct["train"]

    #model.fit(interactions_train, sample_weight=sample_weight_train, epochs=30, verbose=False)
    #scores = model.predict(interactions_valid, cold_start='nan') # 'nan' or 'drop'

    train_users = pd.Series(interactions_train.MEMBER_ID.unique())
    valid_users = pd.Series(interactions_valid.MEMBER_ID.unique())
    valid_users = train_users  # For now, validation set is training set. 
    both_users = set(train_users) & set(valid_users)
    cold_start_users = set(valid_users) - set(train_users)

    # Only used here
    #valid_recs = model.recommend(valid_users, n_items=10, filter_previous=False, cold_start='nan')
    #valid_recs_filter_prev = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='nan')

    # Generate Model Performance Validation Metrics
    #model_hrt = flib.hit_rate(model, interactions_valid, k=topN, max_kept=1, train_interactions=interactions_train)
    print("interactions_valid: ", interactions_valid)
    model_hrt = hit_rate(model, interactions_valid, k=topN, filter_previous=False)
    model_hrt_filter_previous = hit_rate(model, interactions_valid, k=topN, filter_previous=True)
    print("model_hrt: ", model_hrt)
    print("model_filter_previous: ", model_hrt_filter_previous)
    #model_rnk = reciprocal_rank(model, interactions_valid, k=topN)
    #model_pre = precision(model, interactions_valid, k=topN)
    #model_rec = recall(model, interactions_valid, k=topN)

    print("\nModel Metrics")
    print("model hit rate (filter previous): {}".format(round(model_hrt_filter_previous, 3)))
    print("model hit rate (not filter previous): {}".format(round(model_hrt, 3)))
    #print("model reciprocal rank: {}".format(round(model_rnk, 3)))
    #print("model precision: {}".format(round(model_pre, 3)))
    #print("model recall: {}".format(round(model_rec, 3)))

#----------------------------------------------------------------------------------------
def train_valid_dct(dct, train_perc, valid_perc, shuffle=True):
    df_members = dct['df_members']
    dftrain, dfvalid, dftest = train_valid(df_members, train_perc, valid_perc, shuffle=shuffle)
    dct['data_train'] = dftrain
    dct['data_valid'] = dfvalid
    dct['data_test']  = dftest

def train_valid(x, train_perc, valid_perc, shuffle=True):
    """
    Split a dataframe into train, validation, testing setes

    Parameters
    ----------
    x : pandas DataFrame

    train_perc, valid_perc: float
        percent of training data [0,1]
        percent of validation data [0,1]

    shuffle:  [True]
        whether to shuffle the data or not, without replacement

    Notes:
        The first two arguments must satisfy the constraint: (train_perc + valid_perc < 1)

    Return
    ------
    x_train, x_valid, x_test : tuple of train, valid, test dataframes

    """

    nb_el = len(x)
    test_perc = 1. - valid_perc - train_perc;
    perc_valid = valid_perc; n_valid = int(nb_el * perc_valid)
    perc_test = (1.-valid_perc-train_perc); n_test =int(nb_el * perc_test)
    perc_train = train_perc
    if shuffle:
        y = x.sample(frac=1)
    x_valid = y.iloc[0:n_valid]
    x_test = y.iloc[n_valid:n_valid+n_test]
    x_train = y.iloc[n_valid+n_test:]
    return x_train, x_valid, x_test

#-------------------------------------------------------------------------------------------
def restrict_member_attrib(interactions_dct):
    """
    Restrict user_features to members in data_train. The user_features will not 
    change between runs. 

    Return: 
    -------
    user_attrib: DataFrame
        The unique members match the unique members of data_train, a rewquirement of rankfm. 
    """
    print(interactions_dct.keys())
    data_train = interactions_dct['data_train']
    user_attrib = interactions_dct['df_user_attr']
    user_attrib = user_attrib[user_attrib['MEMBER_ID'].isin(data_train.MEMBER_ID)]
    return user_attrib

#----------------------------------------------------------------------------------------------
def run_model(model, interaction_dct, topn=3, verbose=False, nb_epochs=30):
    user_attrib = restrict_member_attrib(interaction_dct)
    data_train = interaction_dct['data_train']
    data_valid = interaction_dct['data_valid']
#     len(set(user_attrib.MEMBER_ID)), len(set(data_valid.MEMBER_ID)), len(set(data_train.MEMBER_ID))
    model.fit(data_train, user_features=user_attrib, sample_weight=None, epochs=nb_epochs, verbose=verbose)
    hr_filtered = hit_rate(model, data_valid, k=topn, filter_previous=True)
    hr_not_filtered = hit_rate(model, data_valid, k=topn, filter_previous=False)
    print("hr (previous filtered): ", hr_filtered)
    print("hr (previous not filtered): ", hr_not_filtered)

#----------------------------------------------------------------------------------------------------
