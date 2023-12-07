# Importar el clasificador Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Datos de ejemplo: opiniones de películas
data = [
    ("Me encantó esta película, fue increíble", "positiva"),
    ("La trama fue confusa y no me gustó", "negativa"),
    ("Actuaciones brillantes, una obra maestra", "positiva"),
    ("No recomendaría esta película a nadie", "negativa"),
    ("Una experiencia aburrida y decepcionante", "negativa"),
    ("¡Una película asombrosa, la recomendaría a todos!", "positiva")
]

# Separar texto y etiquetas
textos = [text for text, _ in data]
etiquetas = [label for _, label in data]


# Vectorización de texto usando CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, etiquetas, test_size=0.3, random_state=42)

# Crear y entrenar el clasificador Naive Bayes Multinomial
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
predictions = classifier.predict(X_test)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, predictions)
print(f"Precisión del clasificador Naive Bayes: {accuracy:.2f}")