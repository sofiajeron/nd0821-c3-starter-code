import pandas as pd
from ml.clean import clean_column_names


# Add code to load in the data.
data = pd.read_csv('../data/census.csv')

#clean_data
data = clean_column_names(data)

#save data
data.to_csv('../data/clean_census.csv', index=False)
