import pandas as pd
import pandas_options
import numpy as np

df_air = pd.read_csv("airports.csv")  
df_cou = pd.read_csv("countries.csv")
df_copa = pd.read_csv("copa_destinations.csv")

print("air: ", df_air.columns)
print("cou: ", df_cou.columns)

# Create a city/country/airport dataframe
df_copa = df_copa[["IATA", "City", "Country"]].sort_values("IATA")
print("\ndf_copa\n", df_copa.head())

# Handle airports.csv
df_air = df_air.drop(["T_l", "T_h"], axis=1)
cols = df_air.columns
df_air.columns = ['IATA'] + list(cols[1:]) # list required. Columns placement unchanged
print("\ndf_air\n", df_air.head())

# Handle countries.csv
df_cou = df_cou.drop(["Case_l", "Case_h"], axis=1)
# Remove space between city and comma
df_cou['City'] = df_cou['City'].str.strip()
print("\ndf_cou\n", df_cou.head())

# Include an IATA column via merge
df = df_cou.merge(df_copa, how="inner", on="City") 
df = df.drop(['Country_x', 'Country_y','City'], axis=1)
cols = list(df.columns)
df = df[[cols[-1]] + cols[0:-1]]  # change column order
print("\n", df.head())

#City,Country,Case_l,avg_yr_l,winter_l,spring_l,summer_l,fall_l,Case_h,avg_yr_h,winter_h,spring_h,summer_h,fall_h

print(df_air.columns)
print(df.columns)
df = pd.concat([df_air, df], axis=0)
print(df)
print(df.shape)
df.columns = ['D'] + list(df.columns[1:])

# GUA appears twice. WHY? 
df.to_csv("airport_temperatures.csv", index=0)
# airports.csv contains GUA twice: 
#GUA,60.16666666666666,57.333333333333336,62.333333333333336,62.0,59.0,75.83333333333333,76.0,78.33333333333333,75.66666666666667,73.33333333333333
#GUA,52.833333333333336,44.66666666666666,57.333333333333336,61.0,48.333333333333336,81.0,79.33333333333333,87.0,79.66666666666667,78.0
# countries.csv: 
# Guatemala City ,guatemala.txt,Low,60.166666666666664,57.333333333333336,62.333333333333336,62.0,59.0,High,75.83333333333333,76.0,78.33333333333333,75.66666666666667,73.33333333333333

# I am missing airports (about 30)
air = set(df['D'])
all_air = set(df_copa['IATA'])
print(len(air), air)
print(len(all_air), all_air)

missing_air = all_air - air
print(len(missing_air), missing_air)

df_copa_missing = df_copa[df_copa['IATA'].isin(missing_air)]
print("df_copa_missing: ", df_copa_missing)

