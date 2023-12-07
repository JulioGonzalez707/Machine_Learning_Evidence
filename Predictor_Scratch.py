import pandas as pd
import numpy as np

# Cargar el archivo CSV
datos = pd.read_csv('Unit_2/Indicadores_municipales_sabana_DA.csv.')

# Convertir los datos de las columnas en arreglos NumPy
N_ic_asalud = np.array(datos['N_ic_asalud'])
Personas_con_carencia = np.array(datos['Personas_con_carencia'])
# Función para calcular los coeficientes de la regresión lineal manualmente
def calcular_coefs(X, y):
    n = np.size(X)
    mean_x, mean_y = np.mean(X), np.mean(y)
    SS_xy = np.sum(y*X) - n*mean_y*mean_x
    SS_xx = np.sum(X*X) - n*mean_x*mean_x
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1*mean_x
    return b_0, b_1

# Función para predecir
def predecir(X, b_0, b_1):
    return b_0 + b_1 * X

# Calculamos los coeficientes
b_0, b_1 = calcular_coefs(N_ic_asalud, Personas_con_carencia)

# Hacemos predicciones
predicciones = predecir(N_ic_asalud, b_0, b_1)
print(predicciones)
