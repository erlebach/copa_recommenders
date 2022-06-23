import numpy as np
import pandas as pd


def series_to_enc_arg(series, enc):
    """
    Translate a series to a one-hot-encoded vector
    """
    return enc.fit_transform( series.to_frame() )

def items_to_raw(series):
    lst = []
    for el in series:
        lst.append(data_training.to_inner_iid(el))
    return np.asarray(lst)
    
def users_to_raw(series):
    lst = []
    for el in series:
        lst.append(data_training.to_inner_uid(el))
    return np.asarray(lst)

def items_to_text_format(series, data, offset=0):
    lst = []
    for i, item in enumerate(series):
        lst.append(f"{data.to_inner_iid(item)+offset}:1")
    #print("lst= ", lst)
    #print("shape: ", np.asarray(lst).shape)
    return np.asarray(lst)

def users_to_text_format(series, data, offset=0):
    lst = []
    for i, user in enumerate(series):
        lst.append(f"{data.to_inner_uid(user)+offset}:1")
    #print("shape: ", np.asarray(lst).shape)
    #print("lst= ", lst)
    return np.asarray(lst)

def create_fn_text_data(data, df):
    n_users = data.n_users
    n_rows = df.shape[0]
    users_text = users_to_text_format(df.userID, data).reshape(-1,1)
    items_text = items_to_text_format(df.itemID, data, offset=n_users).reshape(-1,1)
    labels_text = np.ones(n_rows).reshape(-1,1)
    print(users_text.shape, items_text.shape, labels_text.shape)

    # More generally, dat should be a list of strings. Each line will have a different number of elements
    # because it is in sparse format. 
    dat = np.concatenate([labels_text, users_text, items_text], axis=1)
    return dat

