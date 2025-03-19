import pandas as pd

dataset = pd.read_csv("../dataset/dataset.csv")
print(dataset.head())

print(dataset["AREA NAME"].unique())