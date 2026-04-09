import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Veriyi yükle
df = pd.read_csv("../data/train.csv")

# 2) Girdi ve hedefi ayır
X = df["text"]
y = df["sentiment"]

# 3) Önce train ve temp olarak böl
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4) Temp veriyi validation ve test diye ikiye böl
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# 5) Boyutları yazdır
print("X_train size:", len(X_train))
print("X_val size:", len(X_val))
print("X_test size:", len(X_test))

print("\nTrain label distribution:")
print(y_train.value_counts())

print("\nValidation label distribution:")
print(y_val.value_counts())

print("\nTest label distribution:")
print(y_test.value_counts())