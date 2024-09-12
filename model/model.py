import joblib
from model.preprocess import preprocess_text, remove_stopwords
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

path = "data/data.csv"

df = pd.read_csv(path)

# NOTE: Clean the sentences using some regex
df["clean"] = df["statement"].fillna("").apply(lambda x: preprocess_text(x))

# NOTE: Remove stopwords using nltk database
df["clean"] = df["clean"].apply(lambda x: remove_stopwords(x))

X = df["clean"]
y = df["status"]

# NOTE: Create matrix of TF-IDF features
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = vectorizer.fit_transform(X)

# NOTE: This is the level of regularization in Ridge regression
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

model = LogisticRegression(penalty="l2", max_iter=1000)

# NOTE: 5-fold CV to assess the best C value using accuracy as metric
# to be maximized
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_tfidf, y)

# NOTE: C=1 was found to be the best
best_model = grid_search.best_estimator_

joblib.dump(best_model, "objects/best_model.pkl")
joblib.dump(vectorizer, "objects/vectorizer.pkl")
