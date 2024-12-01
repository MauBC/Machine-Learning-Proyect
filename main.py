import kagglehub
import pandas as pd
import numpy as np
from keras.api.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Función para crear las secuencias de entrenamiento y prueba
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps].values)
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

print("Modelo ya entrenado")
model = load_model('modelo_entrenado.h5')  # Carga el modelo entrenado

# El modelo ya está listo para hacer predicciones o evaluaciones sin necesidad de reentrenar
print("Modelo cargado con éxito")

# Descargar la versión más reciente
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
print(f"Path to dataset files: {path}")

# Cargar el dataset
df = pd.read_csv(path + '/btcusd_1-min_data.csv')
print(df.info())
df.head()

# Convertir Timestamp a datetime
df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')

# Eliminar la columna 'Timestamp'
df = df.drop(columns=['Timestamp'])

# Eliminar filas con valores nulos
df = df.dropna()

# Resamplear por día y calcular el promedio
df_daily = df.resample('D', on='Datetime').mean()

# Seleccionar solo la columna 'Close' como variable objetivo
data = df[['Datetime', 'Close']].set_index('Datetime')

# Crear un objeto MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Aplicar normalización a la columna 'Close'
data['Close'] = scaler.fit_transform(data[['Close']])

# Dividir los datos en entrenamiento (80%) y prueba (20%)
train_size = int(len(data) * 0.8)
test_data = data[train_size:]

# Número de pasos hacia atrás para usar como entrada
n_steps = 60  # Usar los últimos 60 minutos como entrada para predecir el siguiente

# Crear secuencias para entrenamiento y prueba
X_test, y_test = create_sequences(test_data['Close'], n_steps)


#y_pred = model.predict(X_test)

# Guardar las predicciones en un archivo .npy
#np.save('predicciones.npy', y_pred)

#print("Las predicciones se han guardado en 'predicciones.npy'.")

# Cargar las predicciones del modelo guardado
y_pred = np.load('predicciones.npy')
print("Las predicciones se han cargado.")

# Invertir la normalización para interpretar los resultados
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = scaler.inverse_transform(y_pred)

# MÉTRICAS DE REGRESIÓN (Error Cuadrático Medio, Error Absoluto Medio y RMSE)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Convertir a clasificación para usar métricas como Precision, Recall, F1, etc.
# Definir un umbral para clasificación: si el precio siguiente es mayor que el precio anterior, será "1" (aumento), de lo contrario "0" (disminución)
y_test_class = (y_test[1:] > y_test[:-1]).astype(int)  # 1 si el precio sube, 0 si baja
y_pred_class = (y_pred[1:] > y_pred[:-1]).astype(int)  # Lo mismo para las predicciones

# Cálculo de las métricas de clasificación
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
tn, fp, fn, tp = confusion_matrix(y_test_class, y_pred_class).ravel()

# Curva ROC y AUC
fpr, tpr, thresholds = roc_curve(y_test_class, y_pred_class)
roc_auc = auc(fpr, tpr)

# Mostrar las métricas de clasificación
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Tasa de Verdaderos Positivos (TPR): {tp / (tp + fn)}")
print(f"Tasa de Falsos Positivos (FPR): {fp / (fp + tn)}")
print(f"Área bajo la Curva (AUC): {roc_auc}")

# Gráfica de la Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.legend()
plt.show()

# Gráfico de Predicción vs Valores Reales
plt.figure(figsize=(12, 6))

# Graficar los valores reales con un color
plt.plot(y_test, label='Valores Reales', color='blue')

# Graficar las predicciones con otro color
plt.plot(y_pred, label='Predicciones', color='orange')

# Título y etiquetas
plt.title('Predicción vs Valores Reales')
plt.xlabel('Tiempo')
plt.ylabel('Precio de Cierre')

# Agregar leyenda
plt.legend()

# Mostrar la gráfica
plt.show()

# Gráfico de Error de Predicción (Real - Predicho)
error = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.plot(error, label='Error de Predicción', color='red')
plt.title('Error de Predicción (Real - Predicho)')
plt.xlabel('Tiempo')
plt.ylabel('Error')
plt.legend()
plt.show()

# Histograma de los Errores
plt.figure(figsize=(12, 6))
plt.hist(error, bins=50, color='purple', alpha=0.7)
plt.title('Distribución del Error de Predicción')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de Tendencia (Real vs Predicho)
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Valores Reales', color='blue', alpha=0.7)
plt.plot(y_pred, label='Predicciones', color='orange', alpha=0.7)
plt.title('Tendencia de Predicción vs Valores Reales de Bitcoin')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.show()
