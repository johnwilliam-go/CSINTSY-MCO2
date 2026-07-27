import pickle
from pathlib import Path
from typing import List

import pandas as pd
from scipy.sparse import csr_matrix, hstack
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

    x["nonletter word"] = (~x["word"].str.contains(r"^[a-z]+$")).astype(int)

    x["punctuation only"] = (
        x["word"].str.fullmatch(r"[^\w\s]+")
    ).fillna(False).astype(int)
# insert your feature matrix
    return x.iloc[:,1:]

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:

    if not tokens:
        return []
    model_path = Path(__file__).with_name("bernoulli_nb.pkl")
    with model_path.open("rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    handcrafted_features = csr_matrix(
        feature_matrix(tokens).to_numpy()
    )
    ngram_features = vectorizer.transform(tokens)
    combined_features = hstack([
        handcrafted_features,
        ngram_features
    ]).tocsr()
    predictions = model.predict(combined_features)


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

    return predictions.tolist()


if __name__ == "__main__":
    # Example usage
    example_tokens = [
        # Numbers
        "₱500"
    ]

    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print(tags)