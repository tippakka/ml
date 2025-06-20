from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
print(f"X: {X[:5]}, y: {y[:5]}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

print(iris.data.shape)
print(len(X_train))
print(len(y_test))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, pred))

print("Actual class name example:", iris.target_names[2])
y_test_names = [iris.target_names[i] for i in y_test]
pred_names = [iris.target_names[i] for i in pred]
print("Predicted vs Actual")

for i in range(len(pred)):
    print(f"{i + 1}: Predicted = {pred_names[i]}, Actual = {y_test_names[i]}")
