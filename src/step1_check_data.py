import pandas as pd

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

print("TRAIN SHAPE:", train_df.shape)
print("TEST SHAPE:", test_df.shape)

print("\nTRAIN COLUMNS:")
print(train_df.columns)

print("\nFIRST 5 ROWS OF TRAIN:")
print(train_df.head())

print("\nTRAIN LABEL COUNTS:")
if "sentiment" in train_df.columns:
    print(train_df["sentiment"].value_counts())
elif "label" in train_df.columns:
    print(train_df["label"].value_counts())
else:
    print("Label column not found.")