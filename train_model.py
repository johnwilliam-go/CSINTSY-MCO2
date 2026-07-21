import pickle

import pandas as pd
import openpyxl as xl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def feature_matrix(words):
    x = pd.DataFrame({"word": words})
    x["word"] = x["word"].str.lower()

    x["has c,f,j,q,v,x,z"] = x["word"].str.contains(r"[cfjqvxz]").astype(int)

    x["fil prefixes"] = (
        x["word"].str.startswith((
            "pinagka", "pinaka", "pakikipag", "tagapag", "pagkaka", "pakiki", "pagka", "pinag",
            "napaka", "mapag", "pagpa", "ipag", "maka", "maki", "naka", "paki", "pang", "taga",
            "mag", "nag", "pag", "ipa", "ika", "ina", "ka", "pa", "na", "ma", "um", "i"
        ))
    ).astype(int)

    x["fil infixes"] = (
        x["word"].str[1:-1].str.contains(r"(?:um|in)")
    ).astype(int)

    x["fil suffixes"] = (
        x["word"].str.endswith(("an", "in", "han", "hin"))
    ).astype(int)

    x["eng prefix"] = (
        x["word"].str.startswith((
            "inter", "under", "super", "trans", "anti", "over", "post", "auto",
            "non", "pre", "sub", "dis", "mis", "un", "re", "de", "co", "ex", "im",
            "in", "il", "ir"
        ))
    ).astype(int)

    x["eng suffix"] = (
        x["word"].str.endswith((
            "tion", "sion", "ment", "ness", "able", "ible", "ship", "hood", "ward",
            "wise", "less", "full", "ally", "ingly", "ing", "ity", "ive", "ous",
            "est", "ism", "ist", "ize", "ise", "ate", "ence", "ance", "ful", "ous",
            "ial", "ic", "al", "er", "or", "ly", "ed", "es", "s"
        ))
    ).astype(int)

    x["hyphen?"] = (x["word"].str.contains(r"[-]")).astype(int)

    x["fil prefix with hyphen"] = (
            (x["fil prefixes"] == 1) &
            x["word"].str.contains(r"-")).astype(int)

    x["nonletter word"] = (~x["word"].str.contains(r"^[a-z]+$")).astype(int)

    return x.iloc[:, 1:10]




df = pd.read_excel("raw_tokens 20-39.xlsx")


x = pd.DataFrame(df.iloc[:, 3])
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

x["actual"] = pd.DataFrame(df.iloc[:, 4])


x.to_csv("output.csv", index=False)

Y = x.iloc[:, 10]
X = x.iloc[:, 1:10]

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

nb = BernoulliNB()
nb.fit(X_train, Y_train)
nb_preds = nb.predict(X_val)
print("Naive Bayes accuracy:", accuracy_score(Y_val, nb_preds))
print(classification_report(Y_val, nb_preds))

with open("bernoulli_nb.pkl", "wb") as f:
    pickle.dump(nb, f)