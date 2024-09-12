import nltk
from nltk.corpus import stopwords
from nltk.data import find
from nltk.tokenize import word_tokenize
import re
import string


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_square_brackets(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text)


def remove_links(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<.*?>+", "", text)


def remove_punctuation(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}]", "", text)


def remove_newlines(text: str) -> str:
    return re.sub(r"\n", "", text)


def remove_words_with_numbers(text: str) -> str:
    return re.sub(r"\w*\d\w*", "", text)


# HACK: I learned today that I could do this, it's interesting.
def preprocess_text(text: str) -> str:
    functions = [
        to_lowercase,
        remove_square_brackets,
        remove_links,
        remove_html_tags,
        remove_punctuation,
        remove_newlines,
        remove_words_with_numbers,
    ]

    for func in functions:
        text = func(text)

    return text


def download_stopwords_if_needed():
    try:
        find("corpora/stopwords.zip")
    except LookupError:
        nltk.download("stopwords")


def download_punkt_if_needed():
    try:
        find("tokenizers/punkt.zip")
    except LookupError:
        nltk.download("stopwords")


def download_punkt_tab_if_needed():
    try:
        find("tokenizers/punkt_tab.zip")
    except LookupError:
        nltk.download("stopwords")


download_stopwords_if_needed()
download_punkt_if_needed()
download_punkt_tab_if_needed()

stop_words = set(stopwords.words("english"))


def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)
