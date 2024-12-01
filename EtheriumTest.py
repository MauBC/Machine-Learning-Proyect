import os
import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import load_model
import matplotlib.pyplot as plt

# Descargar la versión más reciente
path = kagglehub.dataset_download("prasoonkottarathil/ethereum-historical-dataset")


# Cargar el modelo previamente entrenado
model = load_model('modelo_entrenado.h5')
print("Modelo cargado con éxito")

# Cargar el archivo CSV con datos históricos de Ethereum
# Supongamos que tienes un archivo CSV similar al de Bitcoin con datos de Ethereum
df = pd.read_csv(path + '/ETH_1H.csv')  # Reemplaza con el path correcto a tu archivo

# Mostrar los primeros registros del dataframe para ver cómo está estructurado
print(df.head())
# Asegurarnos de que no haya valores nulos
df = df.dropna()

# Establecer la columna 'Date' como índice
data = df.set_index('Date')

# Si solo te interesa la columna 'Close' para predecir el precio
data = data[['Close']]

# Visualizar las primeras filas
print(data.head())

# Normalizar la columna 'Close'
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

# Verificar los datos normalizados
print(data.head())

# Dividir los datos en entrenamiento y prueba
n_steps = 60  # Usar los últimos 60 minutos como entrada para predecir el siguiente

# Seleccionar los últimos 100 datos como conjunto de prueba (o según lo que desees)
test_data = data[-5000:]  # Cambié de 100 a 5000

# Crear las secuencias de entrada y salida (X_test, y_test)
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps].values)
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(test_data['Close'], n_steps)

# Reshape los datos para que sean compatibles con la entrada del modelo LSTM
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Hacer las predicciones
y_pred = model.predict(X_test)

# Desnormalizar los valores de las predicciones y los valores reales
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)

# Calcular las métricas de error
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Crear la gráfica de predicciones vs valores reales
plt.figure(figsize=(12, 6))

# Graficar los valores reales con un color
plt.plot(y_test, label='Valores Reales', color='blue')

# Graficar las predicciones con otro color
plt.plot(y_pred, label='Predicciones', color='orange')

# Título y etiquetas
plt.title('Predicción vs Valores Reales de Ethereum')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre de Ethereum (USD)')

# Agregar leyenda
plt.legend()

# Mostrar la gráfica
plt.show()