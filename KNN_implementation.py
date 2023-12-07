import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Ejemplo de uso
# Datos de entrenamiento
X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train = np.array([0, 0, 1, 1, 0, 1])

# Datos de prueba
X_test = np.array([[1, 3], [4, 2], [7, 9]])
y_test = np.array([0, 0, 1])

# Crear el modelo KNN
knn = KNN(k=3)
knn.fit(X_train, y_train)

# Realizar predicciones
predictions = knn.predict(X_test)

# Calcular la precisión del modelo
accuracy = np.mean(predictions == y_test)
print(f"Precisión del modelo: {accuracy}")