import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart.csv")

print("Dataset loaded ✅")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained ✅")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

sample = [X_test.iloc[0].values]
print("Prediction:", model.predict(sample))
