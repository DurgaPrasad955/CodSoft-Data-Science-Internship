import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

data = pd.read_csv("IRIS.csv")

print("\nFirst 5 rows of dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=data)
plt.title("Iris Flower Distribution")
plt.show()

X = data[['sepal_length','sepal_width','petal_length','petal_width']]
y = data['species']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

accuracy = accuracy_score(y_test,predictions)

print("\nModel Accuracy:",accuracy)
