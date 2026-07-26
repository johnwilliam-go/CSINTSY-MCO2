import pickle
from typing import List
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

def feature_matrix(words):

    x = pd.DataFrame({"word": words})
    x["word"] = x["word"].str.lower()

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

    return x.iloc[:, 1:11]


df = pd.read_excel("dataset/raw_tokens_annotated_v2.xlsx")
x = feature_matrix(df.iloc[:, 3])
y = df.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

nb = BernoulliNB()
nb.fit(x_train, y_train)
nb_preds = nb.predict(x_test)
print("Naive Bayes accuracy:", accuracy_score(y_test, nb_preds))
print(classification_report(y_test, nb_preds))

with open("bernoulli_nb.pkl", "wb") as f:
    pickle.dump(nb, f)


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    with open('bernoulli_nb.pkl', 'rb') as f:
        model = pickle.load(f)

    features = feature_matrix(tokens)
    predicted = model.predict(features)

    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.

    return [str(tag) for tag in predicted]


if __name__ == "__main__":
    # Example usage
    example_tokens = [
        "Dapat", "nagshift", "na", "lang", "ako", "sa", "mech", "eng", "instead", "of", "cs", ".", "i", "want", "partial", "differential", "equations","."
    ]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print(tags)