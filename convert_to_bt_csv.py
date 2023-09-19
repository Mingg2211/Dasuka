import pandas as pd

df = pd.read_csv('data/__2.csv')
# hscode,mota
def filter_fn(row):
    if len(row['hscode'])<4  and row['mota'][0]!='-':
        return False
    else:
        return True
    
m = df.apply(filter_fn, axis=1)
df1 = df[m]
df1