
# ============================================================
# TASK 2 — MOVIE RATING PREDICTION WITH PYTHON
# CodSoft Data Science Internship
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ LOAD DATASET
data = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("Dataset Shape:", data.shape)
print(data.head())

# 2️⃣ DATA CLEANING

data = data.dropna(subset=["Rating"])

# Clean Year
data["Year"] = data["Year"].astype(str).str.extract(r'(\d{4})')
data["Year"] = pd.to_numeric(data["Year"], errors="coerce")

# Clean Duration
data["Duration"] = data["Duration"].astype(str).str.extract(r'(\d+)')
data["Duration"] = pd.to_numeric(data["Duration"], errors="coerce")

# Clean Votes
data["Votes"] = data["Votes"].astype(str).str.replace(",", "")
data["Votes"] = pd.to_numeric(data["Votes"], errors="coerce")

data = data.dropna()

# 3️⃣ FEATURE ENCODING

le = LabelEncoder()

data["Genre"] = le.fit_transform(data["Genre"])
data["Director"] = le.fit_transform(data["Director"])
data["Actor 1"] = le.fit_transform(data["Actor 1"])
data["Actor 2"] = le.fit_transform(data["Actor 2"])
data["Actor 3"] = le.fit_transform(data["Actor 3"])

# 4️⃣ FEATURES & TARGET

X = data[["Year","Duration","Votes","Genre","Director","Actor 1","Actor 2","Actor 3"]]
y = data["Rating"]

# 5️⃣ TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ MODEL

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# 7️⃣ MODEL EVALUATION

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# ============================================================
# PLOTS
# ============================================================

sns.set_style("darkgrid")

# 📊 Plot 1: Rating Distribution
plt.figure(figsize=(8,4))
sns.histplot(data["Rating"], bins=30, kde=True)
plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# 📊 Plot 2: Actual vs Predicted

plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()
