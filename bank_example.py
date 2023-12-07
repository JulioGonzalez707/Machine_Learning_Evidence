import pandas as pd 

dataset = pd.read_csv('bank-full.csv', sep = ";")

dataset.head(5)

print(dataset.head(5))

print(dataset.describe())

#encouding = latin-1