import pandas as pd
import numpy as np
import utils as u  # from rankfm (poor choice of names, but temporary)

# same function as in rankfm
def hit_rate(model, test_interactions, k=10, filter_previous=False, max_kept=0, train_interactions=None):
    """evaluate hit-rate (any match) wrt out-of-sample observed interactions

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :param max_kept [0]: maximum destinations to keep if found in the training set
    :return: the hit rate or proportion of test users with any matching items
    """

    # ADD COMMENT
    k = k + max_kept

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items dictionary
    test_user_items = pd.DataFrame(u.get_data(test_interactions), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = list(test_user_items.keys())

    # generate topK recommendations for all test users also present in the training data
    # I would like an intermediate case. For example, I would like to keep a maximum of one previous flight
    # So that would require an argument such as keep_previous=1  (keep max of 1 previous destination in validation set)
    test_recs = model.recommend(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    print(test_recs.head())
    # list of users
    comm_user = test_recs.index.values
    # print(len(comm_user), len(test_recs)) # same values

    # Next section should be done inside pyx code, but I do not have access. 
    if filter_previous == False and max_kept == 1:
        assert (train_interactions != None, "train_interactions must be defined")
        train_members = set(train_interactions.index)
        train_interactions = train_interactions.to_frame('Dset')
        #print(train_interactions)
        #print("train_members: ", train_members)
        members = test_recs.index.values
        for member in members:
            ranked_dests = []
            if member in train_members:
                valid_dests = test_recs.loc[member,:].values
                for D in valid_dests:
                    if D in train_interactions.loc[member].Dset:
                        print("gordon")
                        # Need a counter, initialized to zero. Once it reaches max_kept
                        count = 0
                        if counter < max_kept:
                            ranked_dests.append(D)
                        # Allow a number of times equal to max_kept
                    else:
                        ranked_dsts.append(D)
                    pass

                # if len(ranked_dsts) == k, then get out of loop. 
                #print(dests)

    # calculate the hit rate (percentage of users with any relevant recommendation) wrt common users
    #print(test_recs)
    hit_rate = {}
    for kk in range(2):
        hit_rate[kk] = np.mean([int(len(set(test_recs.loc[u]) & test_user_items[u]) > kk) for u in comm_user]) # 2/3 of cpu time
    return hit_rate

#-----------------------------------------------------------------------------------------------
def correct_rate(model, test_interactions, k=10, filter_previous=False):
    """evaluate hit-rate (any match) wrt out-of-sample observed interactions
    # What does "out of sample" mean? 

    :param model: trained RankFM model instance
    :param test_interactions: pandas dataframe of out-of-sample observed user/item interactions
    :param k: number of recommendations to generate for each user
    :param filter_previous: remove observed training items from generated recommendations
    :return: the hit rate or proportion of test users with any matching items
    """

    # ensure that the model has been fit before attempting to generate predictions
    assert model.is_fit, "you must fit the model prior to evaluating hold-out metrics"

    # transform interactions into a user -> items dictionary
    test_user_items = pd.DataFrame(u.get_data(test_interactions), columns=['user_id', 'item_id'])
    test_user_items = test_user_items.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = list(test_user_items.keys())

    # generate topK recommendations for all test users also present in the training data
    test_recs = model.recommend(users=test_users, n_items=k, filter_previous=filter_previous, cold_start='drop')
    comm_user = test_recs.index.values
    # print(len(comm_user), len(test_recs)) # same values

    # calculate the hit rate (percentage of users with any relevant recommendation) wrt common users
    #print(test_recs)
    hit_rate = np.mean([int(len(set(test_recs.loc[u]) & test_user_items[u]) > 0) for u in comm_user])
    return hit_rate

#-----------------------------------------------------------------------------------------------
