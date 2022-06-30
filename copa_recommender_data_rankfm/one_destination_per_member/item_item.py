"""
Author: Gordon Erlebacher
Date: 2022-05-12

Functions used to support the item-item recommender algorithm authored by Desphpande and Karypis in 2004. 
"""

import numpy as np
from collections import defaultdict

def count_zero_items(data, verbose):
    if verbose:
        print("nb users: ", len(data.all_users()))
        print("all ratings: ", len(list(data.all_ratings())))
    count = 0
    total_dest = 0
    for u in data.all_users():
        ln = len(data.ur[u])
        if ln == 0:
            count += 1
        else:
            total_dest += ln
        pass
    print("   total count: ", count)
    if verbose:
        print("total destinations: ", total_dest)

#---------------------------------------------------------
def user_item_row_norms(data):
    """
    Compute row norms of user-item matrix.
    Assumes that all ratings are equal to 1
    """
    row_norms = []  # one per user
    row_norms_inv_sq = []
    for i in data.ur.keys():
        # all ratings are equal to 1
        row_norms.append( np.sqrt(len(data.ur[i])) )
        row_norms_inv_sq.append( 1. / row_norms[-1]**2 )
    row_norms = np.asarray(row_norms)
    row_norms_inv_sq = np.asarray(row_norms_inv_sq)
    return row_norms, row_norms_inv_sq
#---------------------------------------------------------
def user_item_col_norms(data):
    col_norms = []
    col_norms_inv = []
    for i in data.ir.keys():
        col_norms.append( np.sqrt(len(data.ir[i])) )
        col_norms_inv.append( 1. / col_norms[-1] )
    col_norms = np.asarray(col_norms)
    col_norms_inv = np.asarray(col_norms_inv)
    col_norms.shape, col_norms_inv.shape
    return col_norms, col_norms_inv
#---------------------------------------------------------
def user_item_matrix(data):
    """
    Construct user-item matrix (in dense format)
    """
    user_item = np.zeros([data.n_users, data.n_items])
    for user, items in data.ur.items():
        for item, rating in items:
            user_item[user, item] = rating
    return user_item
#---------------------------------------------------------
def user_item_matrix_row_normalized(user_item, row_norms):
    user_item_row = np.copy(user_item)
    n_users = user_item.shape[0]

    for user in range(n_users):
        user_item_row[user, :] /= row_norms[user]

    return user_item_row
#---------------------------------------------------------
def sim_matrix(user_item):
    """
    Similarity Matrix
    """
    n_users, n_items = user_item.shape
    col_norms = np.zeros(n_items)
    for item in range(n_items):
        col_norms[item] = np.linalg.norm(user_item[:, item])

    sim = np.zeros([n_items, n_items])

    for i1 in range(n_items):
        for i2 in range(i1):
            sim[i1, i2] = np.dot(user_item[:, i1], user_item[:, i2]) / (col_norms[i1] * col_norms[i2])
            sim[i2, i1] = sim[i1, i2]

    return sim
#---------------------------------------------------------
def normalize_model_cols(model):
    model2 = model.copy()
    n_items = model.shape[1]

    for j in range(n_items):
        n = np.linalg.norm(model2[:,j])
        if n != 0:
            model2[:,j] = model2[:,j] / n

    return model2
#---------------------------------------------------------
def topN_recommend(model, userId, data,  N):
    """
    Arguments:
    ----------
    model:  (numpy array) of size (nb_items x nb_items)
        model(i,j) contains the similarity metric between items i and j
    userId: id of user for whom we seek recommendations
    data: generated by Surprise library
        data is a class with methods to access the raw data, including converters
    N: number of recommendations to generate

    Return:
    ------
    topN_recommend returnsa list of recommended items sorted from highest to lowest.
    Each item comes with a rating in the form of a tuple.

    The recommendation values can be greater than 1. So it is not clear how to interpret them.
    """
    items = data.ur[userId]
    n_items = data.n_items
    topN = np.zeros(n_items)

    flew_to = []
    for item, _ in items:
        flew_to.append(data.to_raw_iid(item))

    # model * (items purchased by user)
    # dot(model[k, :], user_id[:]) : row of model * items purchased by userI
    for i in range(n_items):  # for each row of model
        for item, rating in items:
            topN[i] += model[i,item] * rating

    # Remove items already purchased (or destinations) by the user
    for item, _ in items:
        topN[item] = 0.

    args = np.argsort(topN)[::-1]
    top_args = args[0:N]

    items = []
    for arg in top_args:
        items.append(data.to_raw_iid(arg))

    topN = topN[args]#[0:N]
    return (topN, items, flew_to)
#---------------------------------------------------------
def get_raw_users(data, year):
    """
    Get list of members in a given year. 
    
    Parameters: 
    -----------
    data (Dictionary) : (key,value) = (year, filtered dataframe)
    year (str) : year filter
    """
    members = set()
    d = data[year]

    for i in range(d.n_users):
        user = d.to_raw_uid(i)
        members.add(user)

    return members
#---------------------------------------------------------
def destination_items(member_list, data, users_common):
    """
    Find destination items flown to by common members
    """
    user_dest_d = defaultdict(list)
    for user in users_common:
        uid = data.to_inner_uid(user)
        items = data.ur[uid]
        for item, _ in items:
            user_dest_d[user].append(data.to_raw_iid(item))
    return user_dest_d
#---------------------------------------------------------
def predictions(members, d, simil_matrix, train_year='2016', test_year='2017', verbose=False):
    #print("train_year:", train_year, ",  test_year: ", test_year, type(test_year))
    #print(members.keys())
    users = members[train_year].intersection(members[test_year])
    sim = simil_matrix[train_year]
    sim1 = normalize_model_cols(sim)
    N = 3
    count_correct = 0
    count_total = 0
    user_dest = destination_items(users, d[test_year], users)

    for userId in list(users):   # users_common
        user_train = d[train_year].to_inner_uid(userId)
        topN, items, train_flew_to = topN_recommend(sim1, user_train, d[train_year], N=N)
        raw_uid = userId
        test_flew_to = user_dest[raw_uid]
        # Intersect predictions with 2017 flights
        correct_pred = set(items).intersection(set(test_flew_to))
        count_total += 1
        if len(correct_pred) > 0:
            count_correct += 1
        if verbose and topN[0] > 0.:
            print(f"{topN[0:N]}, {items}, {train_year}-flights: {train_flew_to}, {test_year}-flights: {test_flew_to}, ==> Correct: {list(correct_pred)}")
            pass

    print(f"Percentage with a correct prediction {test_year} based on {train_year}: {count_correct}/{count_total}: {count_correct / count_total}")
#---------------------------------------------------------
def tst():
    print("gordon")
#---------------------------------------------------------
def predictions_month(members, d, simil_matrix, train_month=15, test_month=16, verbose=False):
    users = members[train_month].intersection(members[test_month])
    sim = simil_matrix[train_month]
    sim1 = normalize_model_cols(sim)
    N = 3
    count_correct = 0
    count_total = 0
    user_dest = destination_items(users, d[test_month], users)

    for userId in list(users):   # users_common
        user_train = d[train_month].to_inner_uid(userId)
        topN, items, train_flew_to = topN_recommend(sim1, user_train, d[train_month], N=N)
        raw_uid = userId
        test_flew_to = user_dest[raw_uid]
        # Intersect predictions with 2017 flights
        correct_pred = set(items).intersection(set(test_flew_to))
        count_total += 1
        if len(correct_pred) > 0:
            count_correct += 1
        if verbose and topN[0] > 0.:
            print(f"{topN[0:N]}, {items}, {train_year}-flights: {train_flew_to}, {test_year}-flights: {test_flew_to}, ==> Correct: {list(correct_pred)}")
            pass

    print(f"Percentage with a correct prediction in month {test_month} based on month {train_month}: {count_correct}/{count_total}: {count_correct / count_total}")
#---------------------------------------------------------
def predictions_month_freq(members, d, simil_matrix, train_month=15, test_month=16, verbose=False):
    users = members[train_month].intersection(members[test_month])
    sim = simil_matrix[train_month]
    sim1 = normalize_model_cols(sim)
    N = 3
    count_correct = 0
    count_total = 0
    user_dest = destination_items(users, d[test_month], users)

    for userId in list(users):   # users_common
        user_train = d[train_month].to_inner_uid(userId)
        topN, items, train_flew_to = topN_recommend(sim1, user_train, d[train_month], N=N)
        raw_uid = userId
        test_flew_to = user_dest[raw_uid]
        # Intersect predictions with 2017 flights
        correct_pred = set(items).intersection(set(test_flew_to))
        count_total += 1
        if len(correct_pred) > 0:
            count_correct += 1
        if verbose and topN[0] > 0.:
            print(f"{topN[0:N]}, {items}, {train_year}-flights: {train_flew_to}, {test_year}-flights: {test_flew_to}, ==> Correct: {list(correct_pred)}")
            pass

    print(f"Percentage with a correct prediction in month {test_month} based on month {train_month}: {count_correct}/{count_total}: {count_correct / count_total}")
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
