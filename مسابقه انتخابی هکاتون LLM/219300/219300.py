from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
X = train_data["abstract"]
y = train_data["category"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("feature_selection", SelectKBest(chi2, k=1000)),
        ("clf", SVC(kernel="linear", probability=True)),
    ]
)
param_grid = {
    "tfidf__max_features": [1000, 2000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 5],
    "tfidf__max_df": [0.5, 0.75],
    "clf__C": [0.1, 1, 10],
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)
y_val_pred = grid_search.predict(X_val)
final_model = grid_search.best_estimator_
X_test = test_data["abstract"]
test_predictions = final_model.predict(X_test)
output_df = pd.DataFrame(test_predictions, columns=["prediction"])
output_df.to_csv("output.csv", index=False)
