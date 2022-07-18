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
def read_data_attributes_single_file(in_file, age_cuts=None, overwrite_cache=False): #, year_train, year_valid)
    """
    in_file (string)
        File containing all records with member and destination attributes. Note that there is 
        no column identification of which columns are which attributes (TODO IN FUTURE)

    age_cuts : list
        Defines the break between different member age categories.

    overwrite_cache: Bool [False]
        If True, overwrite the cache even if present

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
    - I do not remove duplicate MEMBER_ID/D pairs at this stage.  This is only done for the training/validation/testsing files.
    """

    # The leading underscores indicates a private variable by convention
    # Create a dictionary that persists across function invocations
    if not hasattr(read_data_attributes_single_file, '_dct'):
        read_data_attributes_single_file._dct = {'raw_data' : None }
    _dct = read_data_attributes_single_file._dct

    if overwrite_cache:
        read_data_attributes_single_file_dct['raw_data'] = None

    interact_dct = {}

    if age_cuts == None:
        age_cuts = [0, 30, 50, 70, 120]

    # Ensure that the last value of age_cuts is >= 120 to capture all age groups
    if age_cuts[-1] < 100:
        age_cuts.append(120)

    if age_cuts[-1] < 120:
        age_cuts[-1] = 120

    # User and item attributes are across entire dataset. They are restricted to the training set at a later stage.

    # age_at_flight changes for different flights, but we ignore this over a 5-year period
    cols_user_attrib = ['MEMBER_ID','TRUE_ORIGIN_COUNTRY','ADDR_COUNTRY','age_departure']
    # This should be an argument to the function
    cols_user_attrib = ['MEMBER_ID','age_departure']

    attrib_file = in_file
    if isinstance(_dct['raw_data'], pd.DataFrame):
        df_ = _dct['raw_data']
    else:
        df_ = pd.read_csv(attrib_file)
        # Sort data according to flight date
        df_ = df_.sort_values("FLIGHT_DATE")
        _dct['raw_data'] = df_


    # Each member/ID pair should have only a single flight (no not take temporal data into account)

    # This drop_duplicates should not be done at this point, otherwise, I will never be able to predict
    # a destination in the validation set that was present in the original set for a particular member .
    # This explains why I get worse results when I do not filter out previous flights. 
    # df_ = df_.drop_duplicates(['MEMBER_ID','D'])  # already dropped, but let us make sure. <<<< NOT HERE

    # Create categories for member ages
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
def train_valid_dct(dct, train_perc, valid_perc, temporal=False, shuffle=True):
    """
    - Split a dataframe into train, validation, testing sets. 
    - Use the dictionary 'dct' to pass information. 
    - There is no return function. 
    - The DataFrame 'df_members' is assumed to be ordered accordng to 'FLIGHT_DATE' timestamp. 
    - See function 'train_valid' for more detail. 
    """
    df_members = dct['df_members']
    print(df_members.shape)

    dftrain, dfvalid, dftest = train_valid(df_members, train_perc, valid_perc, shuffle=shuffle)
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
def restrict_member_attrib(interactions_dct):
    """
    Restrict user_features to members in data_train. The user_features will not 
    change between runs. 

    Return: 
    -------
    user_attrib: DataFrame
        The unique members match the unique members of data_train, a rewquirement of rankfm. 
    """
    data_train = interactions_dct['data_train']
    user_attrib = interactions_dct['df_user_attr']
    user_attrib = user_attrib[user_attrib['MEMBER_ID'].isin(data_train.MEMBER_ID)]
    return user_attrib

#----------------------------------------------------------------------------------------------
def run_model(model, interaction_dct, topN=3, verbose=False, nb_epochs=30, with_attrib=True):
    user_attrib = restrict_member_attrib(interaction_dct)
    data_train = interaction_dct['data_train']
    data_valid = interaction_dct['data_valid']

    if with_attrib == True:
        model.fit(data_train, user_features=user_attrib, sample_weight=None, epochs=nb_epochs, verbose=verbose)
    else:
        model.fit(data_train, sample_weight=None, epochs=nb_epochs, verbose=verbose)

    hr_filtered = hit_rate(model, data_valid, k=topN, filter_previous=True)
    hr_not_filtered = hit_rate(model, data_valid, k=topN, filter_previous=False)
    print("hr (previous filtered): ", hr_filtered)
    print("hr (previous not filtered): ", hr_not_filtered)

#----------------------------------------------------------------------------------------------------
def recommender(model, interaction_dct, topN=5, keep_nb_members=None):
    """
    Arguments: 
    ---------
    model
    interactions_dct: dictionary
    topN: [5]
        number of elements to keep
    keep_members: [None]
        number of members to consider in validation

    Return
    ------
    pair: list of pairs (MEMBER_ID, D) used to validate

    """
    user_attrib = restrict_member_attrib(interaction_dct)
    data_train = interaction_dct['data_train']
    data_valid = interaction_dct['data_valid']
    all_data = interaction_dct['df_members']
    all_dest= np.asarray(sorted(list(all_data.D.unique())))

    train_dest_sets = data_train.groupby('MEMBER_ID').agg({'D':set})
    valid_dest_sets = data_valid.groupby('MEMBER_ID').agg({'D':set})
    interaction_dct['train_dest_sets'] = train_dest_sets
    interaction_dct['valid_dest_sets'] = valid_dest_sets

    # For each member in the validation set, estimate the scores for all destinations. 
    members = data_valid.MEMBER_ID.unique()  # ndarray of members

    # Access MEMBER_ID via index or column
    valid_dest_sets['MEMBER_ID'] = valid_dest_sets.index
    train_dest_sets['MEMBER_ID'] = train_dest_sets.index

    # Keep a small number of members

    if keep_nb_members != None:
        members = members[0:keep_nb_members]

    pairs = []
    for i, member in enumerate(members):
        for dest in all_dest:
            pairs.append((member, dest))

    pairs = pd.DataFrame(pairs, columns=['MEMBER_ID','D'])
    # pred will be of smaller size than pairs if cold_start is 'drop'
    pred = model.predict(pairs, cold_start='nan')
    pairs['pred'] = pred

    #-------------------

    res1 = pairs.groupby('MEMBER_ID').agg({'D':list, 'pred':list})

    res1['argsort'] = res1['pred'].apply(lambda x: np.argsort(x)[::-1])

    def extract_topN(res1, topn):
        def argsortcolD(row):
            return np.array(row['D'])[row['argsort'][0:topn]]

        def argsortcolPred(row):
            return np.array(row['pred'])[row['argsort'][0:topn]]

        res1['D1'] = res1.apply(argsortcolD, axis=1)
        res1['pred1'] = res1.apply(argsortcolPred, axis=1)
        res2 = res1.drop(['D','pred', 'argsort'], axis=1)  # lists or ndarray:w
        return res2

    res2 = extract_topN(res1, topn=topN)

    # res2 has columns 'D1' and 'pred1': 
    # 'D1' predicted destinations
    # 'scores' predicted scores in ranked drder

#  First issue: even when running 500 epochs, the scores are a mixture of neg and positive. WHY? 
#  I wonder if this is true with the training data? 

    # Calculate hit rate assuming a hit if "at least" one of topN predictions is in the 
    # flights actually flown in the validation set. 
    # Keep all recommended flights even if found in th e # training set. 


    # Do not filter out hits in the training set
    hits = 0
    for member in members: 
        set_valid = valid_dest_sets.loc[member]
        valid_dests = set_valid['D']  # set
        ranked_dests = res2.loc[member, 'D1']  # ndarray
        # Recall that above, I kept topN+2 ranked elements for the non-filtered case
        if valid_dests.intersection(ranked_dests):
            hits += 1

    print("hit rate (without previous filter) = ", hits/len(members))

    # Filter out hits in the training set
    # Inefficient implementation
    # Calculate the number of this when filtered
    # Calculate hit rate assuming a hit if "at least" one of topN predictions is in the 
    # flights actually flown in the validation set. 
    # Remove from the recommended flights, any flights found in the training set. 

    res2 = extract_topN(res1, topn=8*topN)
    #print("res2: ", res2)

    hits = 0
    nb_members = len(members)
    for member in members:
        # print("member: ", member)
        try:
            td = train_dest_sets.loc[member]  # train destinations (set)
        except:
            nb_members -= 1
            continue
        set_valid = valid_dest_sets.loc[member]
        vd = set_valid['D']    # validation destinations (set)
        rp = res2.loc[member, 'D1'] # ranked prediction (ndarray)
        rk = []
        for d in rp:
            if d in td['D']: 
                continue
            rk.append(d)
            if len(rk) == topN:
                break

        if len(rk) != topN:
            print("len(rk): ", len(rk))
        if len(rk) < 5:
            print("==> member: ", member)
            print("    vd = ", vd, "    (validation dest)")
            print("    td= ", td.values, "  (training dest)")
            print("    rk= ", rk[0:10], "   (ranked dest)")
            print("    rp= ", rp, "    (trimmed ranked dest)")
            print("    len(vd): ", len(vd), "   , len(td): ", len(td))     
        if vd.intersection(rk):
            hits += 1

    # Identical to the hit rate built into rankfm! That is great!
    print("hit rate (with previous filter) = ", hits/nb_members)


    return pairs
#---------------------------------------------------------------------------------------------------
