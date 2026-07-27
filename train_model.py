import pickle
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

def feature_matrix(words):

    x = pd.DataFrame({"word": words})
    x["word"] = x["word"].str.lower()
    x["word"] = x["word"].astype(str)

    x["has c,f,j,q,v,x,z"] = x["word"].str.contains(r"[cfjqvxz]").astype(int)

    common_eng_words = (
        "the", "a", "an", "to", "of", "in", "on", "for", "and", "or",
        "but", "with", "my", "your", "his", "her", "it", "this",
        "that", "is", "was", "are", "were", "be", "i", "you", "we", "they"
    )

    x["common english word"] = x["word"].isin(common_eng_words).astype(int)

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

    x["punctuation only"] = (
        x["word"].str.fullmatch(r"[^\w\s]+")
    ).fillna(False).astype(int)
# insert your feature matrix
    return x.iloc[:,1:]
def train_model():
    df = pd.read_excel("raw_tokens 20-39 (2).xlsx", keep_default_na=False)
    words = df.iloc[:, 3].astype(str)
    y = df.iloc[:, 4]
    # 70% train, 15% validation, 15% test
    words_train, words_temp, y_train, y_temp = train_test_split(
        words, y, test_size=0.30, stratify=y, random_state=42
    )

    words_val, words_test, y_val, y_test = train_test_split(
        words_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    X_train_handcrafted = csr_matrix(feature_matrix(words_train).to_numpy())
    X_val_handcrafted = csr_matrix(feature_matrix(words_val).to_numpy())
    X_test_handcrafted = csr_matrix(feature_matrix(words_test).to_numpy())

    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        binary=True,
        lowercase=True
    )

    X_train_ngram = vectorizer.fit_transform(words_train)
    X_val_ngram = vectorizer.transform(words_val)
    X_test_ngram = vectorizer.transform(words_test)

    X_train = hstack([X_train_handcrafted, X_train_ngram]).tocsr()
    X_val = hstack([X_val_handcrafted, X_val_ngram]).tocsr()
    X_test = hstack([X_test_handcrafted, X_test_ngram]).tocsr()

    nb = BernoulliNB()
    nb.fit(X_train, y_train)
    #validation
    val_preds = nb.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, val_preds))
    print(classification_report(y_val, val_preds))
    #Test
    test_preds = nb.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print(classification_report(y_test, test_preds))

    model_data = {
        "model": nb,
        "vectorizer": vectorizer
    }

    with open("bernoulli_nb.pkl", "wb") as f:
        pickle.dump(model_data, f)

if __name__ == "__main__":
    train_model()