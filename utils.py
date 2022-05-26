import pandas as pd
import numpy as np
import date_library as datelib

def clean_dataframe(df0):
    df = df0.copy()
    df = seg_and_true_od_dest_match(df)
    df = keep_only(df, "OPERATING_COMPANY", "CM")
    # Restrict df to itineraries operated by Copa
    # Holding_OD is the section of the itinerary managed by Copa
    df = df[df['TRUE_OD'] == df['HOLDING_OD']]

    # Is the order of the next two lines important? Should not be.
    df = PNRs_with_multiple_MEMBER_IDs(df)
    df = keep_depart_country_is_addr_country(df)

    # Remove itineraries that originate and terminate in the same country
    # There is no way to determine the country of origin of these flights
    df = remove_flights_within_country(df)

    # Compute the number of each party and add a new column "PARTYSZ"

    # Add a column equal to the number of tickets on this PNR
    df = party_size(df)
    # Each triplet (PNR / FlIGHT# / FLIGHT_DATE) should only appear once
    df = one_record_per_family(df)

    df.fillna(-1, inplace=True)
    float_cols = df.columns[df.dtypes == 'float']
    df[float_cols] = df[float_cols].astype('int')   # recommended use of view generates error
    # It would nice to identify the dates automatically
    date_cols = ['ENROLL_DATE','LAST_TIER_CHANGE_DATE', 'BIRTH_DATE', 'ACTIVITY_DATE', 'FLIGHT_DATE', 'BOOKING_DATE', 'TICKET_SALES_DATE']
    for col in date_cols: 
        print(f"process {col}")
        df[col] = datelib.date_to_timestamp(df[col])

    df.iloc[0].T.head(50)
    return df
#-----------------------------------------------------------
def type_distribution(dff, cols=None):
    """
    List the various categories where column names include the word "TYPE", along with countss
    
    Arguments
    ---------
        dff (DataFrame)
    Return
    ------
        no return
    """
    if not isinstance(cols, list):
        cols = ['ACTIVITY_TYPE','TRANSACTION_TYPE','ISSUING_COMPANY_TYPE','OPERATING_COMPANY_TYPE','SALES_CHANNEL_TYPE']

    for col in cols:
        a = dff.groupby(col).size()
        print("-----------------------------------")
        print("nb classes: ", len(a))
        print(a[0:20])
#-----------------------------------------------------------
def location_stats(df, col_name):
    """
    Print out the number of entities encoded in col_name, and return thist list. 
    Entities are either airports (O/D) or regions or countries. 
    """
    col = df[col_name]
    if rex.findall(r'(region|country)', col_name.lower()):
        O = set(col)
        D = set([])
    else:
        O = set(col.str[0:3])
        D = set(col.str[4:7])
    airports = list(O.union(D)) #.sort()
    print(f"{col_name}, nb airports: ", len(airports))
    airports.sort()
    return airports
#-----------------------------------------------------------
def PNRs_with_multiple_MEMBER_IDs(df):
    """
    Ensure 1-1 correspondance between PNR and MEMBER_IDs

    Arguments
    ---------
    df (Pandas DataFrame)
        Should have the columns PNR, MEMBER_ID

    Return
    ------
    (Pandas DataFrame)
        Each PNR is associated with a single MEMBER_ID
    """
    df_members_g = df[['PNR','MEMBER_ID']].groupby(['PNR','MEMBER_ID'])
    df_members_g_df = df_members_g.count().reset_index('MEMBER_ID')

    df_members_g_df_g = df_members_g_df.groupby('PNR')
    size_df = df_members_g_df_g.size().sort_values(ascending=False).to_frame('size')

    PNRsGT1 = size_df[size_df['size'] > 1]
    PNRsEQ1 = size_df[size_df['size'] == 1]

    print(f"nb PNRs with with more than one member: {PNRsGT1.sum()}")
    #print(PNRsGT1)

    # Only keep rows with PNR in the list PNRsEQ1
    return df.set_index('PNR').loc[PNRsEQ1.index].reset_index('PNR')
#-----------------------------------------------------------
def read_activity_file(nrows=None): 
    return pd.read_csv('MEMBERS_ACTIVITY.csv', nrows=nrows, encoding = "ISO-8859-1")
#-----------------------------------------------------------
def find_word(lst, word): 
    lst = list(lst)
    for l in lst:
        if word in l:
            print(l)
    print()
#-----------------------------------------------------------
def keep_depart_country_is_addr_country(df):
    addr = df['ADDR_COUNTRY'].str.lower().str.strip() == df['TRUE_ORIGIN_COUNTRY'].str.lower().str.strip()
    return (df[addr])
#-----------------------------------------------------------
def seg_and_true_od_dest_match(df):
    return df[df['TRUE_OD'].str[4:7] == df['SEGMENT_OD'].str[4:7]]
#-----------------------------------------------------------
def keep_only(df, col, value):
    return df[df[col] == value] 
#-----------------------------------------------------------
def remove_flights_within_country(df):
    return df[df['TRUE_ORIGIN_COUNTRY'] != df['TRUE_DESTINATION_COUNTRY']]
#-----------------------------------------------------------
def one_record_per_family(df):
    """
    Filter the input dataframe to leave only a single ticket per itinerary.
    Every combination of (PNR / FLIGHT_DATE / OPERATING_FLIGHT_NUMBER) should appear once.
    If there were multiple tickets associated with this entry, that would be the number
    of passengers in the party. 

    Arguments
    ---------
    df (Pandas DataFrame)

    Return
    ------
    Filtered Pandas Dataframe, different from the input. 

    Note
    ----
    I am not sure this method is constructed properly. TO DEBUG. 
    """
    cols_drop = ['PNR','FLIGHT_DATE', 'OPERATING_FLIGHT_NUMBER']
    return df.drop_duplicates(subset=cols_drop, keep='first')
#-----------------------------------------------------------
def party_size(df):
    """
    Identify the size of each party. Group by (PNR / FLIGHT_DATE / OPERATING_FLIGHT_NUMBER / TICKET_NUMBER) 
    The size of each entry is the party size. 

    Arguments
    ---------
    df (Pandas DataFrame)

    Return
    ------
    Filtered Pandas Dataframe, different from the input. 

    Note
    ----
    I am not sure this method is constructed properly. TO DEBUG. 
    """
    cols_party = ['PNR','FLIGHT_DATE', 'OPERATING_FLIGHT_NUMBER', 'TICKET_NUMBER']
    dfg = df.groupby(cols_party)
    party_sz = dfg.size()
    max_party_sz = party_sz.max()
    print("max party siz: ", max_party_sz)
    df['PARTY_SZ'] = dfg['TRUE_OD'].transform('size')
    return df
#-----------------------------------------------------------
def get_dayofweek_month(df, df_date_col='date', labels=('date', 'month', 'dayofweek')):
    """
    Add Month and day of week columns to df compouted from date_range

    Arguments
    ---------
    df: DataFrame with a date (string) column
    df_date_col: name of date columns in df
    labels: labels to assign to the month and dayofweek

    Return
    ------
    A dataframe with three string columns: day, month, day of week
    """

    df1 = df.copy()

    # Convert timestamps to dates
    assert(type(df.iloc[0][df_date_col]) == np.int64)

    min_date = df[df_date_col].min()
    max_date = df[df_date_col].max()

    # Convert two timestamps to dates
    min_date = datelib.timestampToDateTimePTY(min_date)[0]
    max_date = datelib.timestampToDateTimePTY(max_date)[0]

    days = pd.date_range(min_date, max_date, freq='D').to_series()   # strings

    day_of_week = days.dt.day_of_week.astype(int)
    date = days.dt.date.astype(str)
    month = days.dt.month.astype(int)

    days_df = pd.DataFrame({labels[0]: date, labels[2]: day_of_week, labels[1]: month}).reset_index(drop=True)
    days_df['dates'] = days_df[labels[0]]
    #print("days_df: ", days_df.columns)

    # create a string date column from timestamp
    df1['dates'] = df[df_date_col].apply(datelib.timestampToDateTimePTY).apply(lambda x: x[0])

    #print("df1: ", df1.columns, df1.FLIGHT_DATE.head())
    #print("days_df: ", days_df.columns, days_df[df_date_col].head())

    df = pd.merge(df1, days_df, how='inner', on='dates')
    #df = pd.merge(df1, days_df, how='inner', left_on='dates', right_on='dates')
    try:
        #print(df['dates'].head())
        df = df.drop('dates', axis=1, inplace=False)
    except:
        print("except")
        pass

    return df
#-----------------------------------------------------------
