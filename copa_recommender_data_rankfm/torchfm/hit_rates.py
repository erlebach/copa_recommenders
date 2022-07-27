import pandas as pd
import numpy as np

#import os
#import sys
import random

from datetime import datetime
#from rankfm.rankfm import RankFM
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall, diversity

#from scipy.sparse import csr_matrix

import function_lib as flib

#--------------------------------------------------------------------------

def evaluate_hit_rate(model, year1, year2, nb_epochs=30, keep_nb_users=None, filter_previous=False, seed=None, verbose=False):
    """
    Arguments: 
        year1, year2: year 1 is the training data, year 2 is the testing data
    Return: :
        Dictionary of arguments and results
    """

    results = {}
    results['keep_nb_users'] = keep_nb_users
    results['filter_previous'] = filter_previous
    results['year1'] = year1
    results['year2'] = year2
    results['nb_epochs'] = year2
    results['seed'] = seed

    if seed != None:
        random.seed(seed)
    else:
        random.seed(datetime.now())
        #print("datetime.now: ", datetime.now())

    #print(f"Evaluate_hit_rate: years: {year1}, {year2}")

    years = [year1, year2]
    df = {}
    files = {}
    for year in years:
        files[year] = f"members_{year}.csv"
        df[year] = pd.read_csv(files[year])

    for year in years:
        df[year] = df[year][['userID', 'itemID']]
        df[year].columns = ['user_id', 'product_id']
        df[year]['year'] = year

    # #### Load Destination Data
    # #### Create a User/Item Interaction Data Set

    df_g = {}
    for year in years:
        df_g[year] = df[year].groupby(['user_id', 'product_id'])

    interactions = pd.concat([df[year] for year in years], axis=0)

    # #### Check that there is only one entry per member_id/destination/year group  

    interactions.groupby(['user_id','product_id','year']).size().max()


    # #### Evaluate User/Item Interaction Sparsity

    n_users = len(interactions.user_id.unique())
    n_items  = len(interactions.product_id.unique())

    sparsity = 1 - (interactions[['user_id', 'product_id']].drop_duplicates().shape[0] / (n_users * n_items))
    #print("full interaction data sparsity: {}".format(round(100 * sparsity, 2)))
    results['sparsity_all'] = sparsity


    all_users = interactions.user_id.unique()
    all_items = interactions.product_id.unique()

    nb_users = len(all_users)

    if keep_nb_users == None:
        keep_nb_users = nb_users - 10

    # Keep a subset of users across all years! (copied from rankfm Instacart demo)
    s_users = np.random.choice(all_users, size=keep_nb_users, replace=False)

    # #### Get All Interactions for Those Users

    s_interactions = interactions[interactions.user_id.isin(s_users)].copy()
    s_interactions.shape

    s_items = s_interactions.product_id.unique()

    interactions.groupby(['user_id', 'product_id']).size()

    # #### Re-Evaluate Cardinality/Sparsity on the Sample

    n_s_users = len(s_users)
    n_s_items = len(s_items)


    s_sparsity = 1 - (s_interactions[['user_id', 'product_id']].drop_duplicates().shape[0] / (n_s_users * n_s_items))
    results['sparsity'] = s_sparsity

    if verbose:
        print("sample users:", n_s_users)
        print("sample items:", n_s_items)
        print("sample interactions:", s_interactions.shape)
        print("sample interaction data sparsity: {}".format(round(100 * s_sparsity, 2)))

    shuffle_index = np.arange(len(s_interactions))
    np.random.shuffle(shuffle_index)

    s_interactions = s_interactions.iloc[shuffle_index]
    s_interactions['random'] = np.random.random(size=len(s_interactions))

    # #### Define training and validation data
    # My data contains data from two years. Use previous year as training, following year as testing
    # test_pct = 0.25
    # GE: this trick is not necessary since s_interactions was subjected to a random permutation
    train_mask = s_interactions['year'] == year1
    valid_mask = s_interactions['year'] == year2
    # train_mask = s_interactions['random'] <  (1 - test_pct)
    # valid_mask = s_interactions['random'] >= (1 - test_pct)

    # Shuffle full file (training + validation). Not sure why. 
    interactions_total = s_interactions[['user_id', 'product_id']]
    interactions_total = interactions_total.iloc[shuffle_index]

    interactions_train = s_interactions[train_mask].groupby(['user_id', 'product_id']).size().to_frame('orders').reset_index()
    interactions_valid = s_interactions[valid_mask].groupby(['user_id', 'product_id']).size().to_frame('orders').reset_index()

    #display(interactions_train.head())

    # Not sure why this does not work with a very low number of samples
    sample_weight_train = np.log2(interactions_train['orders'] + 1)
    sample_weight_valid = np.log2(interactions_valid['orders'] + 1)
    
    interactions_train = interactions_train[['user_id', 'product_id']]
    interactions_valid = interactions_valid[['user_id', 'product_id']]

    #train_mask.sum(), valid_mask.sum(), len(train_mask), len(valid_mask), len(interactions_total), len(interactions_train), len(interactions_valid)
    #sample_weight_train.max(), sample_weight_valid.max()

    train_users = np.sort(interactions_train.user_id.unique())
    valid_users = np.sort(interactions_valid.user_id.unique())
    cold_start_users = set(valid_users) - set(train_users)  #  6 ms
    # cold1 = set(interactions_valid.user_id[~interactions_valid.user_id.isin(interactions_train.user_id)])  # 3 ms
    # print("len(cold1): ", len(cold1))

    train_items = np.sort(interactions_train.product_id.unique())
    valid_items = np.sort(interactions_valid.product_id.unique())
    cold_start_items = set(valid_items) - set(train_items)
    # print(len(cold_start_users)) # 0
    #print(len(cold_start_items)) # 1959

    # item_features_train = item_features[item_features.product_id.isin(train_items)]
    # item_features_valid = item_features[item_features.product_id.isin(valid_items)]

    if verbose:
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

    # ### Test Out Core Package Functionality
    # #### Initialize the Model with Chosen Hyperparameters

    #### Fit the Model on the Training Data and Profile Computational Performance

    # FIT crashes with an insufficient number of samples (100 samples not enough. WHY?
    # Talk to Nathan about this. Why a kernel error with an insufficient number of samples? And 
    # why do I not see the printouts? Because of buffering? 
    #print("sample_weight_train: ", sample_weight_train.max())
    #print(interactions_train.head(50))
    #print("before return", flush=True)
    #print(help(model.fit))
    model.fit(interactions_train, sample_weight=sample_weight_train, epochs=nb_epochs, verbose=verbose)

    #### Generate Model Scores
    scores = model.predict(interactions_valid, cold_start='nan') # 'nan' or 'drop'


    # #### Generate TopN Recommendations

    train_users = pd.Series(interactions_train.user_id.unique())
    valid_users = pd.Series(interactions_valid.user_id.unique())
    both_users = set(train_users) & set(valid_users)
    cold_start_users = set(valid_users) - set(train_users)

    # ### Evaluate Model Performance on the Validation Data
    # #### Specify Number of Recommended Items

    k = 3

    results['topN=k'] = k

    # Create dictionary: member => list of items
    # Works even when there are there are multiple entries for the same (user_id, product_id)
    # 1/3 sec
    test_user_items = interactions_valid.groupby('user_id')['product_id'].apply(set)   # .to_dict()
    if len(test_user_items) == 0:
        print("There are no test users")
        return

    """
    # #### Generate Pure-Popularity Baselines
    # This allows the popular flights to also occur in the training data.
    most_popular = interactions_train.groupby('product_id')['user_id'].count().sort_values(ascending=False)[:k]
    base_hrt = np.mean([int(len(set(most_popular.index) & set(val)) > 0)                       for key, val in test_user_items.items()])
    base_pre = np.mean([len(set(most_popular.index) & set(val)) / len(set(most_popular.index)) for key, val in test_user_items.items()])
    base_rec = np.mean([len(set(most_popular.index) & set(val)) / len(set(val))                for key, val in test_user_items.items()])

    results['most_popular'] = most_popular
    results['base_hrt'] = base_hrt
    results['base_pre'] = base_pre
    results['base_rec'] = base_rec
    """

    """
    print("\nPure-Popularity Baselines")
    print("number of test users: {}".format(len(test_user_items)))
    print("baseline hit rate: {:.3f}".format(base_hrt))
    print("baseline precision: {:.3f}".format(base_pre))
    print("baseline recall: {:.3f}".format(base_rec))
    """

    # I should provide the same reuslts by removing the popular flights that are in the training set. For this, use k=15, and 
    # remove the flights in the training set. TODO


    # #### Generate Model Performance Validation Metrics

    """
    #model_hrt = hit_rate(model, interactions_valid, k=k, filter_previous=filter_previous)
    model_rnk = reciprocal_rank(model, interactions_valid, k=k, filter_previous=filter_previous)
    model_pre = precision(model, interactions_valid, k=k, filter_previous=filter_previous)
    model_rec = recall(model, interactions_valid, k=k, filter_previous=filter_previous)

    results['model_rnk'] = model_rnk
    results['model_pre'] = model_pre
    results['model_rec'] = model_rec
    """

    """
    print("\nModel Performance Validation Metrics")
    print("model hit rate: {}".format(round(model_hrt, 3)))
    print("model reciprocal rank: {}".format(round(model_rnk, 3)))
    print("model precision: {}".format(round(model_pre, 3)))
    print("model recall: {}".format(round(model_rec, 3)))
    """

    # average number of users for which at least one recommendation is correct
    # 77% hit rate! (at least one hit correct)
    # 26.6% hit rate (at least two hits correct)
    # 5.6% (three hits correct)
 
    #model_hrt_orig = hit_rate(model, interactions_valid, filter_previous=filter_previous, k=k)  # same function as in rankfm
    model_hrt = flib.hit_rate(model, interactions_valid, filter_previous=filter_previous, k=k)  # same function as in rankfm
    results['model_hrt'] = model_hrt # returns a dictionary
    #for kk in range(2):
        ##print(f"model hit rate (average nb hits > {kk}): {model_hrt[kk]}")
        #pass

    return results

#-----------------------------------------------------------------------------------------------------------------------------------------------
