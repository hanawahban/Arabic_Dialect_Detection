import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from src.models import run_all_experiments

df = pd.read_excel("data/CleanedData.xlsx")

print("Original shape:", df.shape)
print("Columns:", df.columns)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

#drop missing values
df = df.dropna(subset=["Sentence", "Dialect"]).copy()

print("\nShape after removing missing values:", df.shape)

# dialect distribiution and sentence length stats before cleaning and outlier removal
print("\nDialect distribution before outlier removal:")
print(df["Dialect"].value_counts())


df["sentence_length"] = df["Sentence"].astype(str).apply(len)

print("\nSentence length stats before cleaning/outlier removal:")
print(df["sentence_length"].describe())

#cleaning and normalization
df["Sentence"] = df["Sentence"].astype(str)

df["Sentence"] = df["Sentence"].str.replace("إ", "ا", regex=False)
df["Sentence"] = df["Sentence"].str.replace("أ", "ا", regex=False)
df["Sentence"] = df["Sentence"].str.replace("آ", "ا", regex=False)
df["Sentence"] = df["Sentence"].str.replace("ى", "ي", regex=False)
df["Sentence"] = df["Sentence"].str.replace("ؤ", "و", regex=False)
df["Sentence"] = df["Sentence"].str.replace("ئ", "ي", regex=False)
df["Sentence"] = df["Sentence"].str.replace("ة", "ه", regex=False)

# remove non-Arabic characters, punctuation, and numbers
df["Sentence"] = df["Sentence"].str.replace(r"[^\u0600-\u06FF\s]", " ", regex=True)

# Clean extra spaces
df["Sentence"] = df["Sentence"].str.strip()
df["Sentence"] = df["Sentence"].str.replace(r"\s+", " ", regex=True)


df["sentence_length"] = df["Sentence"].apply(len)

# remove very short texts
df = df[df["sentence_length"] >= 3].copy()

df["sentence_length"] = df["Sentence"].apply(len)

#remove the 1% longest sentences 
upper_limit = df["sentence_length"].quantile(0.99)
df = df[df["sentence_length"] <= upper_limit].copy()

df["sentence_length"] = df["Sentence"].apply(len)

#results post cleaning 
print("\nShape after removing outliers and short texts:", df.shape)

print("\nSentence length stats after removing outliers and missing values:")
print(df["sentence_length"].describe())

print("\nDialect distribution after removing outliers and missing values:")
print(df["Dialect"].value_counts())

print("\nUpper limit used for outlier removal:", upper_limit)

#train/test split
X = df["Sentence"]
y = df["Dialect"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

#word tf-idf
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_word = word_vec.fit_transform(X_train)
X_test_word = word_vec.transform(X_test)

print("\nWord TF-IDF shape:", X_train_word.shape)

#character tf-idf
char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=5000
)

X_train_char = char_vec.fit_transform(X_train)
X_test_char = char_vec.transform(X_test)

print("Char TF-IDF shape:", X_train_char.shape)

os.makedirs("results", exist_ok=True)

run_all_experiments(
    X_train_word, X_test_word,
    X_train_char, X_test_char,
    y_train, y_test
)