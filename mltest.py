from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

prediction = model.predict([X[0]])

print("Prediction:", prediction[0])
print("Actual:", y[0])
