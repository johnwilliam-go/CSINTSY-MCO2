import pandas as pd
import openpyxl as xl
import numpy as np

df = pd.read_excel("raw_tokens.xlsx")


x = pd.DataFrame(df.iloc[344:617, 3])
x["word"] = x["word"].str.lower()


x["has c,f,j,q,v,x,z"] = x["word"].str.contains(r"[cfjqvxz]").astype(int)


x["fil prefixes"] = (
    x["word"].str.startswith((
        "pinagka","pinaka","pakikipag","tagapag","pagkaka","pakiki","pagka","pinag",
        "napaka","mapag","pagpa","ipag","maka","maki","naka","paki","pang","taga",
        "mag","nag","pag","ipa","ika","ina","ka","pa","na","ma","um","i"
    ))
    ).astype(int)


x["fil infixes"] = (
    x["word"].str[1:-1].str.contains(r"(?:um|in)")
    ).astype(int)


x["fil suffixes"] = (
    x["word"].str.endswith(("an","in","han","hin"))
    ).astype(int)


x["eng prefix"] = (
    x["word"].str.startswith((
        "inter","under","super","trans","anti","over","post","auto",
        "non","pre","sub","dis","mis","un","re","de","co","ex","im",
        "in","il","ir"
    ))
    ).astype(int)


x["eng suffix"] = (
    x["word"].str.endswith((
        "tion","sion","ment","ness","able","ible","ship","hood","ward",
        "wise","less","full",   "ally","ingly","ing","ity","ive","ous",
        "est","ism","ist","ize","ise","ate","ence","ance","ful","ous",
        "ial","ic","al","er","or","ly","ed","es","s"
    ))
    ).astype(int)

x["hyphen?"] = (x["word"].str.contains(r"[-]") ).astype(int)

x["fil prefix with hyphen"] = (
    (x["fil prefixes"] == 1) &
    x["word"].str.contains(r"-")).astype(int)

x["nonletter word"] = (~x["word"].str.contains(r"^[a-z]+$")).astype(int)

print(x.iloc[1,0])
x.to_csv("output.csv", index=False)
print(x)