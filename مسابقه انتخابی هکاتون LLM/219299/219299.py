import pandas as pd
import re
from collections import Counter
import emoji


def find_most_common_words(text, n=5):
    # words = re.findall(r"\b[a-zA-Z]+\b", text, re.IGNORECASE)
    words = text.split()
    counts = Counter(words)
    return counts.most_common(n)


def find_unique_words(text): 
    # words = re.findall(r"\b[a-zA-Z]+\b", text, re.IGNORECASE)
    words = text.split()
    counts = Counter(words)
    return [word for word, count in counts.items() if count == 1]

qoura_df = pd.read_csv("qoura_questions.csv")
text_data = " ".join(qoura_df["question"].tolist())
words = re.findall(r"\b[m][a-z]{2,}[t]\b", text_data, re.IGNORECASE)

emoji_count = sum(1 for char in text_data if emoji.is_emoji(char))

most_common_words = find_most_common_words(text_data)

unique_words = find_unique_words(text_data)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(f"{len(words)}\n")
    f.write(f"{emoji_count}\n")
    f.write(
        " ".join(["{}:{}".format(word, count) for word, count in most_common_words])
        + "\n"
    )
    f.write(f"{len(unique_words)}\n")
