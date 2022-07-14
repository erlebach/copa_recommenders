# Library to support rankfm

import pandas as pd
import numpy as np
import sys
import pandas_options

from rankfm.rankfm import RankFM
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall, diversity
import function_lib as flib

def read_data(year_train, year_valid):
    """
    year_train (int)
        Year to use for training data
    year_valid (int)
        Year to use as validation data

    Return
    ------
    df (DataFrame)
       u Dictionary with a minimum of four columns (userID (member), itemID (Destination), rating, year))
        The year column is the year of booking
    """
    df = {}
    files = {}
    years = [year_train, year_valid]
    for year in years:
        files[year] = f"members_{year}.csv"
        df[year] = pd.read_csv(files[year])

    for year in years:
        df[year] = df[year][['userID', 'itemID']]
        df[year].columns = ['MEMBER_ID', 'D']
        df[year]['year'] = year

    return df
#----------------------------------------------------------------------------------------
def read_data_attributes(year, year_train, year_valid):
    """
    year_train (int)
        Year to use for training data
    year_test (int)
        Year to use as testing data

    Return
    ------
    df (DataFrame)
       u Dictionary with a minimum of four columns (userID (member), itemID (Destination), rating, year))
        The year column is the year of booking
    """

    attribs_nonbit = f"attrib_{year}.csv"
    years = [year_train, year_valid]
    df = {}
    df_members = {}
    df_user_attrib = {}
    # age_at_flight changes for different flights, but we ignore this over a 5-year period
    cols_user_attrib = ['MEMBER_ID','TRUE_ORIGIN_COUNTRY','ADDR_COUNTRY','age_departure']

    for year in years:
        attrib_file = f"attrib_{year}.csv"
        df[year] = pd.read_csv(attrib_file)
        # Each member/ID pair should have only a single flight (no not take temporal data into account)
        df[year] = df[year].drop_duplicates(['MEMBER_ID','D'])
        # Bin passenger ages
        df[year]['age_departure'] = pd.cut(df[year].age_at_flight, bins=[0, 30, 50, 70, 120])
        df_members[year] = df[year][['MEMBER_ID', 'D']]
        df_user_attrib[year]  = df[year][cols_user_attrib].drop_duplicates('MEMBER_ID')
        # Keep all columns; one-hot encode the specified columns
        df_user_attrib[year] = pd.get_dummies(df_user_attrib[year], prefix=['toc','ac','aaf'],
          columns=['TRUE_ORIGIN_COUNTRY','ADDR_COUNTRY','age_departure'])

    # Read item attributes
    df1 = pd.read_csv("temp_massaged.csv")
    # Initial experimentation: with min/max avg temperatures during the year
    df1 = df1.drop(["avg_wi", "avg_sp", "avg_su", "avg_fa"], axis=1)
    # Attributes will be considered as scalars. Divide by 100 for normalization
    df1['avg_yr_l'] = df1.avg_yr_l / 100.
    df1['avg_yr_h'] = df1.avg_yr_h / 100.
    df_item_attr = df1.copy()
    return df_members, df_user_attrib, df_item_attr

#----------------------------------------------------------------------------------
def sparsity(interactions):
    n_users = len(interactions.MEMBER_ID.unique())
    n_items  = len(interactions.D.unique())
    sparsity = 1 - (interactions[['MEMBER_ID', 'D']].drop_duplicates().shape[0] / (n_users * n_items))
    dct = {'sparsity':sparsity, 'n_users':n_users, 'n_items':n_items}
    return dct
#-----------------------------------------------------------------------------------
def shuffle_interaction_data(s_interactions):
    shuffle_index = np.arange(len(s_interactions))
    np.random.shuffle(shuffle_index)
    shuffled_interactions = s_interactions.copy().iloc[shuffle_index].reset_index()
    shuffled_interactions.drop('index', axis=1, inplace=True)
    return shuffled_interactions, shuffle_index
#----------------------------------------------------------------------------
def train_validation(s_interactions, year1, year2, shuffle_index):
    # My data contains data from two years. Use previous year as training; use following year as testing
    # GE: this trick is not necessary since s_interactions was subjected to a random permutation
    train_mask = s_interactions['year'] == year1
    valid_mask = s_interactions['year'] == year2

    # Shuffle full file (training + validation). Not sure why.
    interactions_total = s_interactions[['MEMBER_ID', 'D']]
#     print(interactions_total.shape)
    interactions_total = interactions_total.iloc[shuffle_index]

    # Create 'orders' column
    interactions_train = s_interactions[train_mask].groupby(['MEMBER_ID', 'D']).size().to_frame('orders').reset_index()
    interactions_valid = s_interactions[valid_mask].groupby(['MEMBER_ID', 'D']).size().to_frame('orders').reset_index()
    #display(interactions_train.head())
    # sample_weight_train = interactions_train['orders']
    # sample_weight_valid = interactions_valid['orders']

    # I will not take the sample_weight into account
    sample_weight_train = np.log2(interactions_train['orders'] + 1)
    sample_weight_valid = np.log2(interactions_valid['orders'] + 1)

    print(interactions_train.columns)
    interactions_train = interactions_train[['MEMBER_ID', 'D']]
    interactions_valid = interactions_valid[['MEMBER_ID', 'D']]

    print(train_mask.sum(), valid_mask.sum(), len(train_mask), len(valid_mask),
          len(interactions_total), len(interactions_train), len(interactions_valid))

    interactions_dct = {
            "train": interactions_train, 
            "valid": interactions_valid,
            "total": interactions_total,
            "sample_weight_train": sample_weight_train,
            "sample_weight_valid": sample_weight_valid,
    }

    return interactions_dct
#--------------------------------------------------------------------
def print_stats(interactions_dct):
    interactions_train = interactions_dct["train"]
    interactions_valid = interactions_dct["valid"]
    interactions_total = interactions_dct["total"]
    sample_weight_train = interactions_dct["sample_weight_train"]
    sample_weight_valid = interactions_dct["sample_weight_valid"]

    train_users = np.sort(interactions_train.MEMBER_ID.unique())
    valid_users = np.sort(interactions_valid.MEMBER_ID.unique())
    print(f"{len(train_users)} train_users, {len(valid_users)} valid_users")
    cold_start_users = set(valid_users) - set(train_users)  #  6 ms
    print("nb cold start users: ", len(cold_start_users))
    # cold1 = set(interactions_valid.MEMBER_ID[~interactions_valid.MEMBER_ID.isin(interactions_train.MEMBER_ID)])  # 3 ms
    # print("len(cold1): ", len(cold1))

    # Product ID is destination
    train_items = np.sort(interactions_train.D.unique())
    valid_items = np.sort(interactions_valid.D.unique())
    cold_start_items = set(valid_items) - set(train_items)
    print("nb cold start users: ", len(cold_start_users)) # 0
    print("nb cold start items: ", len(cold_start_items)) # 1959

    # item_features_train = item_features[item_features.D.isin(train_items)]
    # item_features_valid = item_features[item_features.D.isin(valid_items)]

    if interactions_total:  # if not None
        print("total shape: {}".format(interactions_total.shape))

    print("train shape: {}".format(interactions_train.shape))
    print("valid shape: {}".format(interactions_valid.shape))

    print("\ntrain weights shape: {}".format(sample_weight_train.shape))
    print("valid weights shape: {}".format(sample_weight_valid.shape))

    print("\nnb train users: {}".format(len(train_users)))
    print("nb valid users: {}".format(len(valid_users)))
    print("nb cold-start users: {}".format(len(cold_start_users)))

    print("\ntrain items: {}".format(len(train_items)))
    print("valid items: {}".format(len(valid_items)))
    print("number of cold-start items: {}".format(len(cold_start_items)))
    print("cold start items: ", cold_start_items)

    # print("\ntrain item features: {}".format(item_features_train.shape))
    # print("valid item features: {}".format(item_features_valid.shape))
#---------------------------------------------------------------------------------------
def topn_recommendations(model, interactions_dct, topN=3):
#def topn_recommendations(model, interactions_dct, base_rec, topN=3):
    interactions_train = interactions_dct["train"]
    interactions_valid = interactions_dct["valid"]
    interactions_total = interactions_dct["total"]

    #model.fit(interactions_train, sample_weight=sample_weight_train, epochs=30, verbose=False)
    #scores = model.predict(interactions_valid, cold_start='nan') # 'nan' or 'drop'

    train_users = pd.Series(interactions_train.MEMBER_ID.unique())
    valid_users = pd.Series(interactions_valid.MEMBER_ID.unique())
    both_users = set(train_users) & set(valid_users)
    cold_start_users = set(valid_users) - set(train_users)

    # Only used here
    #valid_recs = model.recommend(valid_users, n_items=10, filter_previous=False, cold_start='nan')
    #valid_recs_filter_prev = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='nan')

    # Pure popularity baselines
    k = topN

    """
    most_popular = interactions_train.groupby('D')['MEMBER_ID'].count().sort_values(ascending=False)[:topN]

    test_user_items = interactions_valid.groupby('MEMBER_ID')['D'].apply(set)   # .to_dict()

    base_hrt = np.mean([int(len(set(most_popular.index) & set(val)) > 0)                       for key, val in test_user_items.items()])
    base_pre = np.mean([len(set(most_popular.index) & set(val)) / len(set(most_popular.index)) for key, val in test_user_items.items()])

    print("\nTest User Items")
    print(test_user_items)
    """

    # Print popularity metrics
    #print("\nPopularity Metrics")
    #print("number of test users: {}".format(len(test_user_items)))
    #print("baseline hit rate: {:.3f}".format(base_hrt))
    #print("baseline precision: {:.3f}".format(base_pre))
    #print("baseline recall: {:.3f}".format(base_rec))

    # Generate Model Performance Validation Metrics
    #model_hrt = flib.hit_rate(model, interactions_valid, k=topN, max_kept=1, train_interactions=interactions_train)
    model_hrt = hit_rate(model, interactions_valid, k=topN, filter_previous=False)
    model_hrt_filter_previous = hit_rate(model, interactions_valid, k=topN, filter_previous=True)
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
"""
def all_in_one():
    s_interactions = interactions[interactions.MEMBER_ID.isin(s_users)].copy()
    shuffled_interactions, shuffle_index = rfmlib.shuffle_interaction_data(s_interactions)
interactions_dct = rfmlib.train_validation(shuffled_interactions, 2016, 2017, shuffle_index)
    # max_samples=500 creates problem for 'warp', but not for 'bpr'. Or vce-versa. What is the difference? And Why?
# max_samples: nb negative samples
    model = RankFM(factors=50, loss='warp', max_samples=50, alpha=0.01, learning_rate=0.1, learning_schedule='invscaling')

"""
#--------------------------------------------------------------------------------------
def topn_recommendations_with_attributes(model, interactions_dct, user_attributes, topN=3):
#def topn_recommendations_with_attributes(model, interactions_dct, user_attributes, base_rec, topN=3):
    interactions_train = interactions_dct["train"]
    interactions_valid = interactions_dct["valid"]
    interactions_total = interactions_dct["total"]

    #model.fit(interactions_train, sample_weight=sample_weight_train, epochs=30, verbose=False)
    #scores = model.predict(interactions_valid, cold_start='nan') # 'nan' or 'drop'

    train_users = pd.Series(interactions_train.MEMBER_ID.unique())
    valid_users = pd.Series(interactions_valid.MEMBER_ID.unique())
    both_users = set(train_users) & set(valid_users)
    cold_start_users = set(valid_users) - set(train_users)

    # Only used here
    #valid_recs = model.recommend(valid_users, n_items=10, filter_previous=False, cold_start='nan')
    #valid_recs_filter_prev = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='nan')

    # Pure popularity baselines
    """
    k = topN
    most_popular = interactions_train.groupby('D')['MEMBER_ID'].count().sort_values(ascending=False)[:topN]

    test_user_items = interactions_valid.groupby('MEMBER_ID')['D'].apply(set)   # .to_dict()

    base_hrt = np.mean([int(len(set(most_popular.index) & set(val)) > 0)                       for key, val in test_user_items.items()])
    base_pre = np.mean([len(set(most_popular.index) & set(val)) / len(set(most_popular.index)) for key, val in test_user_items.items()])
    """

    #print("\nTest User Items")
    #print(test_user_items)

    # Print popularity metrics
    #print("\nPopularity Metrics")
    #print("number of test users: {}".format(len(test_user_items)))
    #print("baseline hit rate: {:.3f}".format(base_hrt))
    #print("baseline precision: {:.3f}".format(base_pre))
    #print("baseline recall: {:.3f}".format(base_rec))

    # Generate Model Performance Validation Metrics
    #model_hrt = flib.hit_rate(model, interactions_valid, k=topN, max_kept=1, train_interactions=interactions_train)
    model_hrt = hit_rate(model, interactions_valid, k=topN, filter_previous=False)
    model_hrt_filter_previous = hit_rate(model, interactions_valid, k=topN, filter_previous=True)
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
def calculate_dct_with_attributes(nb_samples=10000, with_attrib=False, verbose=False, nb_epochs=30, loss='warp'): 
    """
    Self-contained function: reads the data and calculates a hit rate
    """
    train_year = 2016
    valid_year  = 2017
    years = [train_year, valid_year]

    # Dictionaries of (MEMBER_ID, D) and member attributes (by year)
    member_dest_df, df_user_attrib, df_item_attrib = read_data_attributes(2016, train_year, valid_year)

    #print("df_item_attrib.shape: ", df_item_attrib.shape, len(set(df_item_attrib.D)))
    df_item_attrib = df_item_attrib.reset_index(drop=True)
    #print(df_item_attrib.sort_values('D'))  # 91 values

    # Any shuffling and filteirng operation on member_dest_df implies that df_user_attrib must be filtered
    # to make sure it does not contain members possibly deleted from member_dest_df

    df = member_dest_df.copy()

    # Do not shuffle
    interactions = pd.concat([df[year] for year in years], axis=0)
    interactions.shape, interactions.columns

    dct = sparsity(interactions)

    # Subsample data for initial testing

    all_users = interactions.MEMBER_ID.unique()
    keep_nb_users = len(all_users)
    if nb_samples < keep_nb_users:
        keep_nb_users = nb_samples

    np.random.seed(1492)
    # keep_nb_users = nb_users   # Keep all the users
    # shuffle the users
    s_users = np.random.choice(all_users, size=keep_nb_users, replace=False)

    # Filter users in df_item_onebot
    for year in years:
        df_user_attrib[year] = df_user_attrib[year][df_user_attrib[year]['MEMBER_ID'].isin(s_users)]

    # Filter users in member_dest_df
    for year in years:
        member_dest_df[year] = member_dest_df[year][member_dest_df[year]['MEMBER_ID'].isin(s_users)]
        assert member_dest_df[year].groupby(['MEMBER_ID', 'D']).size().max() == 1, "max flights by one passenger per year should be 1"

    # I can weight the samples in various ways. I am not doing so. 
    sample_weight_train = np.ones(member_dest_df[train_year].shape[0])
    sample_weight_valid = np.ones(member_dest_df[train_year].shape[0])

    interactions_dct = {
            "train": member_dest_df[train_year], 
            "valid": member_dest_df[valid_year], 
            "total": None, #interactions_total,
            "sample_weight_train": sample_weight_train,
            "sample_weight_valid": sample_weight_valid,
            "user_attrib_train": df_user_attrib[train_year],
            "user_attrib_valid": df_user_attrib[valid_year], # not used normally
            "item_attrib": df_item_attrib,
    }

    print_stats(interactions_dct)

    interactions = pd.concat([df[year] for year in years], axis=0)
    interactions.shape, interactions.columns

    interactions_train = interactions_dct["train"]
    user_features = interactions_dct["user_attrib_train"]
    item_features = interactions_dct["item_attrib"]
    sample_weight_train = interactions_dct["sample_weight_train"]

    if loss != 'warp' and loss != 'bpr':
        raise("loss must be 'warp' or 'bpr'")

    model = RankFM(factors=50, loss=loss, max_samples=50, alpha=0.01, learning_rate=0.1, learning_schedule='invscaling')
    #model = RankFM(factors=50, loss='warp', max_samples=50, alpha=0.01, learning_rate=0.1, learning_schedule='invscaling')

    # Handle item (Flight Destination) features
    print("=============================")
    #print("item_features\n: ", item_features)
    print("interactions_train\n: ", interactions_train.shape)
    print("item_featues\n: ", item_features.shape)

    #user_feat = user_features['MEMBER_ID'].unique()
    #train_feat = interactions_train['MEMBER_ID'].unique()

    # Drop Destinations from item_features not contained in interactions_train
    dest_feat = set(interactions_train['D'])
    item_feat = set(item_features['D'])
    #print("len dest_feat: ", len(dest_feat))
    #print("len item_feat: ", len(item_feat))
    items_to_remove = dest_feat - item_feat
    #print(">>>> items_to_remove: ", items_to_remove)  # REC
    item_features = item_features[item_features['D'].isin(dest_feat)].reset_index(drop=True)

    #print("unique D features : ", len(dest_feat))
    #print("unique D in training data : ", len(set(interactions_train.D)))
    #print("unique item features: ", len(item_features['D'].unique()))
    #print("remaining ID in training data: ", dest_feat - set(item_features['D']))
    #print(item_features)

    # Remove duplicates from item_features (item 25  occurs twice. WHY?)
    item_features = item_features.drop_duplicates('D')

    if with_attrib:
        #model.fit(interactions_train, user_features=user_features, item_features=None, sample_weight=sample_weight_train, epochs=nb_epochs, verbose=verbose)
        model.fit(interactions_train, user_features=user_features, item_features=item_features, sample_weight=sample_weight_train, epochs=nb_epochs, verbose=verbose)
    else:
        model.fit(interactions_train, sample_weight=sample_weight_train, epochs=nb_epochs, verbose=verbose)

    # Generate model scores
    interactions_valid = interactions_dct["valid"]
    scores = model.predict(interactions_valid, cold_start='nan') # 'nan' or 'drop'`

    # Does not change model
    #valid_recs = model.recommend(valid_users, n_items=10, filter_previous=False, cold_start='nan')
    #valid_recs = model.recommend(valid_users, n_items=10, filter_previous=True, cold_start='nan')

    #interactions_dct['model'] = model

    return interactions_dct, model
#----------------------------------------------------------------------------------------
"""
def all_in_one():
    s_interactions = interactions[interactions.MEMBER_ID.isin(s_users)].copy()
    shuffled_interactions, shuffle_index = rfmlib.shuffle_interaction_data(s_interactions)
interactions_dct = rfmlib.train_validation(shuffled_interactions, 2016, 2017, shuffle_index)
    # max_samples=500 creates problem for 'warp', but not for 'bpr'. Or vce-versa. What is the difference? And Why?
# max_samples: nb negative samples
    model = RankFM(factors=50, loss='warp', max_samples=50, alpha=0.01, learning_rate=0.1, learning_schedule='invscaling')

"""
