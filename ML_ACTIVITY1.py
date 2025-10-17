from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris(as_frame=True)
df = iris.frame
print(df.head())


print(df.describe())
print("Target classes:", iris.target_names)