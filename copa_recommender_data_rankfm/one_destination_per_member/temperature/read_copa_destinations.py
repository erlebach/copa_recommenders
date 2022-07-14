# Read a file that contains Copa destinations
# Produce a data frame with the cities
# Add the min/max temperatures
# Make the destination the first column
# Use the file as item features

# List of cities copy/pasted from 
# https://en.wikipedia.org/wiki/List_of_Copa_Airlines_destinations

# Sunshine hours in cities around the world by month 
# https://en.wikipedia.org/wiki/List_of_cities_by_sunshine_duration

# Average monthly precipitation in European cities
# https://en.wikipedia.org/wiki/List_of_cities_in_Europe_by_precipitation

# Average yearly precipitation in US cities
# https://www.currentresults.com/Weather/US/average-annual-precipitation-by-city.php

# Climate of 100 selected US cities
# https://www.infoplease.com/math-science/weather/climate-of-100-selected-us-cities
# Avg monthly T (Jan, April, July, Oct), Avg annual precipitation (in)/(days), avg inch snowfall

# Yearly average rainfall Latin America Cities
# https://www.currentresults.com/Weather/South-America/Cities/precipitation-annual-average.php

# https://www.visualcrossing.com/resources/documentation/weather-data-tutorials/how-can-i-download-weather-history-data-as-a-csv-file/
# https://www.visualcrossing.com/weather/weather-data-services#
#  Looks good. I should download 2 years of data for 80 cities. 

import pandas as pd
import numpy as np
import pandas_options

filename = 'copa_destinations.csv'
df = pd.read_csv(filename)

# Manually removed the two Airports from the input file: 
#   Mariscal Sucre International Airport (closed) (SEQU in ICAO column)
#   Felipe Angeles International Airport (starts Sept. 22) (MMSM in ICAO column)

# Put destination column at far left
D = df['IATA']
df = df.drop(['IATA','ICAO','Airport','Refs'], axis=1)
df['D'] = D
cols = list(df.columns)
df = df[ ['D'] + cols[0:-1] ]

print(df)

#----------------------------------------------------------
# Read temperature file, and merge with city file

from temperature import mean_temperatures

print(mean_temperatures.shape, df.shape)
print(mean_temperatures)
print(df)

df_cities = pd.merge(df, mean_temperatures, how='inner', on='City')
print("====> ")
print(df_cities)

dest_with_temperatures = set(df_cities['D'].values)
all_dest = set(D.values)

dest_without_temperatures = all_dest - dest_with_temperatures
print(len(dest_without_temperatures))
print(sorted(list(dest_without_temperatures)))
