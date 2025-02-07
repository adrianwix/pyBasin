from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Training data
X_train = np.array([[1, 0], [0, 1]])
y_train = np.array([1, 2])

# Create and train the kNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Example usage
X_test = np.array([
    [1, 0],  # Label 1
    [0, 1],  # Label 2
    [1, 0],
    [0, 1],
    [0, 1],
])
y_pred = knn.predict(X_test)

print("Predicted labels:", y_pred)
