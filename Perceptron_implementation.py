import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Inicializar pesos y bias a cero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Entrenamiento del perceptrón
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                # Función de pérdida (Loss function) - Perceptrón simple
                loss = max(0, -y[i]*linear_output)
                
                # Actualización de pesos y bias
                if loss != 0:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = np.sign(linear_output)
        return y_predicted

    def activation(self, x):
        return np.where(x >= 0, 1, -1)

# Ejemplo de uso
# Datos de entrenamiento
X_train = np.array([[2, 3], [1, 2], [3, 4], [5, 1]])
y_train = np.array([1, 1, -1, -1])

# Datos de prueba
X_test = np.array([[4, 5], [2, 1]])
y_test = np.array([-1, 1])

# Crear el perceptrón
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)

# Entrenar el perceptrón
perceptron.fit(X_train, y_train)

# Realizar predicciones
predictions = perceptron.predict(X_test)

# Calcular la precisión del modelo
accuracy = np.mean(predictions == y_test)
print(f"Precisión del modelo: {accuracy}")