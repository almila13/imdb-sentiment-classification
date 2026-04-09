import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1) Veriyi yükle
df = pd.read_csv("../data/train.csv")

# 2) Girdi ve hedef
X = df["text"]
y = df["sentiment"]

# 3) Train / validation / test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# 4) TF-IDF tanımla
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

# 5) Sadece train üzerinde fit et
X_train_tfidf = vectorizer.fit_transform(X_train)

# 6) Validation ve test'i aynı vectorizer ile transform et
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# 7) Kontrol için yazdır
print("X_train_tfidf shape:", X_train_tfidf.shape)
print("X_val_tfidf shape:", X_val_tfidf.shape)
print("X_test_tfidf shape:", X_test_tfidf.shape)

print("\nİlk 20 feature ismi:")
print(vectorizer.get_feature_names_out()[:20])