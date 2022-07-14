import pandas as pd
import pandas_options

df = pd.read_csv("city_temperature.csv")
print(df.columns)
print(df.shape)

# Drop all rows with negative temperature (missing data)
df = df[df['AvgTemperature'] > 0]

# Calculate min/max years for all Country/Cities
cities = ['Country','State','City']
cities = ['City']
minmax_year = df.groupby(cities).agg({'Year': ['min','max']})
#print(minmax_year)
df['min_year'] = df.groupby(cities)['Year'].transform('min')
df['max_year'] = df.groupby(cities)['Year'].transform('max')
df['min_year'] = df['max_year'] - 5
print(df.head())
print(df.shape)

# Only keep row where year is between min and max year

# Average temperatures between min and max year
df2 = df[(df['Year'] >= df['min_year'])]
print(df2.shape)

# Min and Max temperature of all cities 
minT = df2.groupby('City')['AvgTemperature'].min().to_frame('minT')
maxT = df2.groupby('City')['AvgTemperature'].max().to_frame('maxT')
avgT = df2.groupby('City')['AvgTemperature'].mean().to_frame('avgT')

print(minT.shape, maxT.shape, avgT.shape)

mean_temperatures = pd.concat([minT,avgT,maxT], axis=1)
print(mean_temperatures)


