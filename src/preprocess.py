from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(messages):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(messages)
    return X, vectorizer
