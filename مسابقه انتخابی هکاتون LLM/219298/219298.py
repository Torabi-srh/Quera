import pandas as pd

qoura_questions_df = pd.read_csv("qoura_questions.csv")
shereno_df = pd.read_csv("shereno.csv")
qoura_questions_df["unique_words"] = qoura_questions_df["question"].apply(
    lambda x: len(set(x.split()))
)
total_unique_words = qoura_questions_df["unique_words"].sum()
qoura_digits_count = (
    qoura_questions_df["question"].apply(lambda x: sum(c.isdigit() for c in x)).sum()
)
shereno_digits_count = (
    shereno_df["Poem"].apply(lambda x: sum(c.isdigit() for c in x)).sum()
)
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())


def count_stopwords(text, stopwords):
    return sum(word in stopwords for word in text.split())


shereno_df["stopwords_count"] = shereno_df["Poem"].apply(
    lambda x: count_stopwords(x, stopwords)
)
total_stopwords_count = shereno_df["stopwords_count"].sum()
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(f"{total_unique_words}\n")
    f.write(f"{qoura_digits_count} {shereno_digits_count}\n")
    f.write(f"{total_stopwords_count}\n")
