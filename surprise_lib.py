import pandas as pd
import numpy as np
import date_library as dlib
from surprise import Dataset, Reader
import item_item as ii

def dates_year(df, year):
    """
    Return all rows of the given year (2nd arg)

    Parameters:
    ----------
    df (Dataframe) : dataframe to filter
    year (string) : select all rows of the given year
    """
    first = year + '-01-01'
    last = year + '-12-31'
    return df[(df['date'] >= first) & (df['date'] <= last)]

def months_filter(df):
    """
    Add two columns: year and month, and a third, which is a month counter
    
    Parameters:
    ----------
    df: dataframe to filter by month/year
    
    Return: 
    -------
    No return. But the dataframe argument is modified.
    """
    df['year'] = df['date'].str[0:4].astype('int')
    df['month'] = df['date'].str[5:7].astype('int')
    min_year = df['year'].min()
    df['month_index'] = (df.year - min_year) * 12 + df.month


#-------------------------------------------------------------------------------
def create_simil_matrix(d, years, r_norms):
    simil_matrix = {}
    user_item = {}
    user_item_row_normalized = {}
    simil_matrix_row_normalized = {}

    for year in years:
        user_item[year] = ii.user_item_matrix(d[year])
        simil_matrix[year] = ii.sim_matrix(user_item[year])
        user_item_row_normalized[year] = ii.user_item_matrix_row_normalized(user_item[year], r_norms[year])
        simil_matrix_row_normalized[year] = ii.sim_matrix(user_item_row_normalized[year])
    return simil_matrix
#-------------------------------------------------------------------------------
def create_simil_matrix_month(d, months, r_norms):
    simil_matrix = {}
    user_item = {}
    user_item_row_normalized = {}
    simil_matrix_row_normalized = {}

    for mo in months:
        user_item[mo] = ii.user_item_matrix(d[mo])
        simil_matrix[mo] = ii.sim_matrix(user_item[mo])
        user_item_row_normalized[mo] = ii.user_item_matrix_row_normalized(user_item[mo], r_norms_mo[mo])
        simil_matrix_row_normalized[mo] = ii.sim_matrix(user_item_row_normalized[mo])
    return simil_matrix
#-------------------------------------------------------------------------------
def members_by_year(df, years):
    # MemberID per year
    """
    members = {}
    for year in years:
        members[year] = ii.get_raw_users(d, year)
        # print(f"Year: {year}, nb members: {len(members[year])}")
    """
    # Much faster version of the four lines above
    members_per_year_df = df.groupby('year')['userID'].apply(set).to_frame('user_list')
    members_per_year_df['user_list'] = members_per_year_df['user_list'].apply(list)
    # year index is an 'int'

    members = {}
    for year in years:
        # for some reason, when apply values, I get an array with a list as its only element
        members[year] = members_per_year_df.loc[int(year)].values # Set
        members[year] = set(members[year][0])
    # members dictionary takes a string argument
    return members
#--------------------------------------------------------------------------------------
def members_by_month(df, months):
    # MemberID per year
    # Much faster version of the four lines above
    members_per_month_df = df.groupby('month_index')['userID'].apply(set).to_frame('user_list')
    members_per_month_df['user_list'] = members_per_month_df['user_list'].apply(list)
    # year index is an 'int'

    members = {}
    for mo in months:
        # for some reason, when apply values, I get an array with a list as its only element
        members[mo] = members_per_month_df.loc[int(mo)].values # Set
        members[mo] = set(members[mo][0])
    # members dictionary takes a string argument
    return members
#------------------------------------------------------------------------------------

# handle years

def create_years(max_rows=None):
    if max_rows == None:
        df = pd.read_csv("member_d.csv")
    else:
        df = pd.read_csv("member_d.csv", nrows=max_rows)

    df = df[['MEMBER_ID','D','FLIGHT_DATE','PARTY_SZ']]
    df.columns = ['userID', 'itemID', 'flight_date', 'family_size']
    max_rating = 5
    #df['rating'] = df['rating'].clip(lower=0., upper=max_rating)
    df['rating'] = 1
    #reader = Reader(rating_scale=[1, max_rating])   # All ratings are 1
    #data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    dates = df.flight_date.values
    # Date routines work properly in both directions
    # it is not clear why the max date is Dec. 2022? That should not be!
    #print("dates.min: ", dlib.timestampToDateTimePTY(dates.min()))
    #print("dates.max: ", dlib.timestampToDateTimePTY(dates.max()))

    new_dates = []
    for date in dates:
        d = dlib.timestampToDateTimePTY(date)[0]
        new_dates.append(d)

    # Choose dates in 2016
    df['date'] = new_dates
    df = df.sort_values('flight_date')

    months_filter(df)

    df_years = {}
    # An iterator can only be used once
    years = list(map(str, range(2015, 2023))) # does not include 2023
    for year in years:
        df_years[year] = dates_year(df, year)
        #print(f"year: {year}, nb records: {df_years[year].shape}")

    # Make sure that every (userID, itemID) pair occurs only once
    dff = {}
    for year in years:
        dff[year] = df_years[year].groupby(['userID','itemID']).size().to_frame('rating').reset_index()
        dff[year]['rating'] = 1 # Do not take nb of trips to a destination into account (YET)
        #print(f"{year}: {dff[year].shape[0]} rows")
        dff[year].to_csv(f"members_{year}.csv", index=0)

    #  Process through Reader from Surprie library
    reader = Reader(rating_scale=[1, 1])

    data_training = {}
    for year in years:
        data_training[year] = Dataset.load_from_df(dff[year][['userID', 'itemID', 'rating']], reader)
        #print(f"Raw ratings: {data_training[year].df.shape[0]}")
    
    d = {}
    for year in years:
        d[year] = data_training[year].build_full_trainset()
        #print(f"{year}: {len(d[year].ur.keys())}, {len(d[year].ir.keys())}")

    # Construct user-item matrix simil_matrix as a numpy array
    r_norms = {}; r_norms_inv_sq = {}
    c_norms = {}; c_norms_inv = {}
    for year in years:
        r_norms[year], r_norms_inv_sq[year] = ii.user_item_row_norms(d[year])
        c_norms[year], c_norms_inv[year] = ii.user_item_col_norms(d[year])

    simil_matrix = create_simil_matrix(d, years, r_norms)

    # MemberID per year
    """
    members = {}
    for year in years:
        members[year] = ii.get_raw_users(d, year)
        # print(f"Year: {year}, nb members: {len(members[year])}")
    """

    members = members_by_year(df, years)

    for i1 in years[:-1]:
        i1 = int(i1)
        i2 = i1 + 1
        ii.predictions(members, d, simil_matrix, train_year=str(i1), test_year=str(i2), verbose=False)
    
    return df, dff, d

#----------------------------------------------------------------------------------------
def create_months(df, max_rows=None):
    df = df.sort_values('flight_date')

    # An iterator can only be used once
    months = list(range(12,61))

    months_filter(df)

    df_months = {}
    dfg = df.groupby('month_index')
    for mo in months:
        df_months[mo] = dfg.get_group(mo)

    dff = {}
    for mo in months:
        dff[mo] = df_months[mo].groupby(['userID','itemID']).size().to_frame('rating').reset_index()

    # Make sure that every (userID, itemID) pair occurs only once
    dff = {}
    for mo in months:
        dff[mo] = df_months[mo].groupby(['userID','itemID']).size().to_frame('rating').reset_index()
        dff[mo]['rating'] = 1 # Do not take nb of trips to a destination into account (YET)

    #  Process through Reader from Surprie library
    reader = Reader(rating_scale=[1, 1])

    data_training = {}
    for mo in months:
        data_training[mo] = Dataset.load_from_df(dff[mo][['userID', 'itemID', 'rating']], reader)
    
    d = {}
    for mo in months:
        d[mo] = data_training[mo].build_full_trainset()

    # Construct user-item matrix simil_matrix as a numpy array
    r_norms = {}; r_norms_inv_sq = {}
    c_norms = {}; c_norms_inv = {}
    for mo in months:
        r_norms[mo], r_norms_inv_sq[mo] = ii.user_item_row_norms(d[mo])
        c_norms[mo], c_norms_inv[mo] = ii.user_item_col_norms(d[mo])

    simil_matrix = create_simil_matrix(d, months, r_norms)

    # Much faster version of the four lines above
    members = members_by_month(df, months)

    for i1 in months[:-1]:
        i2 = i1 + 1
        ii.predictions_month(members, d, simil_matrix, train_month=i1, test_month=i2, verbose=False)
    
    return df, dff, d

#----------------------------------------------------------------------------------------
def create_months_freq(d_mo, freq=2, max_rows=None):
    """
    Given a monthly subdivision, create subdivision `freq` months at a time. 
    """
    #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    keys = list(d_mo.keys())
    nb_bins = len(keys)

    #  Process through Reader from Surprie library
    reader = Reader(rating_scale=[1, 1])

    data_training = {}
    dffbins = {}

    for n in range(0, nb_bins, freq):
        mo = keys[n]

        # ONLY WORKS with frewq = 2 for now. For more generality, must correct pd.concat( .... )
        try:
            d_list = [d_mo[mo+i] for i in range(freq)]
            #dffbins[n // freq] = pd.concat([d_mo[mo], d_mo[mo+1]], axis=0)[['userID','itemID','rating']]
            dffbins[n // freq] = pd.concat(d_list, axis=0)[['userID','itemID','rating']]
            data_training[n // freq] = Dataset.load_from_df(dffbins[n // freq], reader)
        except:
            #print("except")
            continue

    nb_bins = len(data_training)
    keys = list(data_training.keys())  # EMPTY

    d = {}
    for key in data_training.keys():
        d[key] = data_training[key].build_full_trainset()

    # Construct user-item matrix simil_matrix as a numpy array
    r_norms = {}; r_norms_inv_sq = {}
    c_norms = {}; c_norms_inv = {}
    for key in data_training.keys():
        r_norms[key], r_norms_inv_sq[key] = ii.user_item_row_norms(d[key])
        c_norms[key], c_norms_inv[key] = ii.user_item_col_norms(d[key])

    simil_matrix = create_simil_matrix(d, keys, r_norms)

    members = {}
    for key in keys:
        members[key] = set(list(dffbins[key].loc[:, "userID"].values))

    # members[key] is a set
    #members = members_by_bin(df, keys)
    print("member keys: ", list(members.keys()))  # WRONG

    for i1 in keys[:-1]:
        i2 = i1 + 1
        ii.predictions_month_freq(members, d, simil_matrix, train_month=i1, test_month=i2, verbose=False)

    return 0, 1, 1

#----------------------------------------------------------------------------------------
def destination_stats(dff):
    """
    Min and max number of destinations for passengers
    Assert that all passengers have at least one destination
    """
    g = dff.groupby('userID')['itemID'].apply(list).to_frame('dest_list')
    min_nb_dest = g['dest_list'].apply(len).min()
    max_nb_dest = g['dest_list'].apply(len).max()
    print("Member with smallest number of destinations: ", min_nb_dest)
    print("Member with largest number of destinations: ", max_nb_dest)
    g['nb_dest'] = g['dest_list'].apply(len)
    assert min_nb_dest > 0, "min nb of destinations must be non-zero"
    print(g[g['nb_dest'] == max_nb_dest])
#-------------------------------------------------------------------------------------------
