import pandas as pd
import openpyxl as xl
import numpy as np

df = pd.read_excel("raw_tokens.xlsx")

words = df.iloc[0:100, 3]

x = pd.DataFrame(df.iloc[0:100, 3])
x["word"] = words

x["has c,f,j,q,v,x,z"] = x["word"].str.contains(r"[cfjqvxz]").astype(int)

print(x)