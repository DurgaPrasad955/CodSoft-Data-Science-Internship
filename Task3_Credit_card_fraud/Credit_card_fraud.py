# ============================================================
# TASK 5 — CREDIT CARD FRAUD DETECTION
# CodSoft Data Science Internship
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ LOAD DATASET

data = pd.read_csv("creditcard.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# 2️⃣ CHECK CLASS DISTRIBUTION

print("\nFraud vs Normal Transactions:")
print(data["Class"].value_counts())

# 3️⃣ FEATURE & TARGET

X = data.drop("Class", axis=1)
y = data["Class"]

# 4️⃣ SCALE AMOUNT COLUMN

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# 5️⃣ TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ MODEL TRAINING

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# 7️⃣ PREDICTIONS

predictions = model.predict(X_test)

# 8️⃣ MODEL EVALUATION

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ============================================================
# PLOTS
# ============================================================

sns.set_style("darkgrid")

# 📊 Plot 1: Class Distribution

plt.figure(figsize=(6,4))
sns.countplot(x=data["Class"])
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# 📊 Plot 2: Confusion Matrix

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()