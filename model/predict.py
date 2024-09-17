import joblib
from model.preprocess import preprocess_text, remove_stopwords

best_model = joblib.load("objects/best_model.pkl")
vectorizer = joblib.load("objects/vectorizer.pkl")

# NOTE: In both functions, I had to add this if statement to check for
# None, since it was giving me an error whenever I tried to start a new
# chat with blank input


def predict_status(sentence: str) -> str:
    if not sentence:
        return "How are you feeling today?"

    clean_sentence = preprocess_text(sentence)
    clean_sentence = remove_stopwords(clean_sentence)

    sentence_tfidf = vectorizer.transform([clean_sentence])

    prediction = best_model.predict(sentence_tfidf)

    return f"Looks like {prediction[0]} Thoughts..."


def predict_proba(sentence: str) -> dict:
    if not sentence:
        return {}

    clean_sentence = preprocess_text(sentence)
    clean_sentence = remove_stopwords(clean_sentence)

    sentence_tfidf = vectorizer.transform([clean_sentence])

    probabilities = best_model.predict_proba(sentence_tfidf)

    classes = best_model.classes_

    prob_dict = dict(zip(classes, probabilities[0]))

    return prob_dict
