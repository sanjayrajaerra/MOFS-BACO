import pandas as pd

df = pd.read_csv("yeast.csv")

classes = list(set(df.iloc[:,-1]))
print(len(classes),len(df))

for i in range(len(df)):
    df.iloc[i,-1] = classes.index(df.iloc[i,-1])

print(df)
df.to_csv("yeast.csv",index=None)