import pandas as pd
import pandas_options
import numpy as np

temp_df = pd.read_csv("airport_temperatures.csv")
print(temp_df.columns)

# Perhaps I should have kept the averages in winter rather than high low. But for now, 
# simply average high and the low. 

temp_df['avg_wi'] = 0.5 * (temp_df.winter_l + temp_df.winter_h)
temp_df['avg_sp'] = 0.5 * (temp_df.spring_l + temp_df.spring_h)
temp_df['avg_su'] = 0.5 * (temp_df.summer_l + temp_df.summer_h)
temp_df['avg_fa'] = 0.5 * (temp_df.fall_l + temp_df.fall_h)

temp_df = temp_df.drop(['winter_l','winter_h','fall_l','fall_h','spring_l','spring_h','summer_l','summer_h'], axis=1)
#print(temp_df)

# Write to "temp_massaged.csv"
temp_df.to_csv("temp_massaged.csv", index=0, float_format='%.0f')

# How to use these temperatures. Start by adding two features to the cities: the temperatures in the four seasons. 
# Ideally, we need to link the travel date to the season.
# Issue: discontinuity between seasons. A trip could start in one season and end in the next season (minority of trips)
