import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/train.csv")

X = df["text"]
y = df["sentiment"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_and_evaluate(optimizer_name="adam"):
    model = MLP(input_dim=10000)
    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    else:
        raise ValueError("Unknown optimizer")

    epochs = 5

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                val_preds.extend(preds.cpu().numpy().flatten())
                val_true.extend(batch_y.cpu().numpy().flatten())

        val_acc = accuracy_score(val_true, val_preds)
        print(f"{optimizer_name.upper()} Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")

    model.eval()
    test_preds = []
    test_true = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            test_preds.extend(preds.cpu().numpy().flatten())
            test_true.extend(batch_y.cpu().numpy().flatten())

    test_acc = accuracy_score(test_true, test_preds)
    print(f"\n{optimizer_name.upper()} Test Accuracy: {test_acc:.4f}\n")
    return test_acc

adam_acc = train_and_evaluate("adam")
sgd_acc = train_and_evaluate("sgd")

print("Final Comparison")
print("Adam Test Accuracy:", adam_acc)
print("SGD Test Accuracy:", sgd_acc)