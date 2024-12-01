from funcs import *

# Download latest version
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")

print("Path to dataset files:", path)
files = os.listdir(path)
print(files)

pd.set_option('display.max_columns', None)
#%matplotlib inline
sns.set_context('notebook')
sns.set_style('whitegrid')
sns.set_palette('Blues_r')

# turn off warnings for final notebook
warnings.filterwarnings('ignore')

# load dataset
df = pd.read_csv(path + '/btcusd_1-min_data.csv')
print(df.info())
df.head()

# Convertir Timestamp a datetime
df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')

# Revisar el dataframe
print(df.head())
print(1)
df = df.drop(columns=['Timestamp'])

# Revisar valores nulos
print(df.isnull().sum())
print(2)
# Eliminar filas con valores nulos (si es necesario)
df = df.dropna()

# O puedes rellenar valores nulos con un método como forward fill
# df.fillna(method='ffill', inplace=True)

print(df.describe())
print(3)
# Graficar el precio de cierre a lo largo del tiempo
plt.figure(figsize=(12, 6))
plt.plot(df['Datetime'], df['Close'], label='Precio de Cierre', color='blue')
plt.title('Evolución del Precio de Cierre')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.show()

# Resamplear por día y calcular el promedio
df_daily = df.resample('D', on='Datetime').mean()

# Graficar el precio de cierre diario
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily['Close'], label='Precio de Cierre (Diario)', color='green')
plt.title('Precio de Cierre Diario')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.show()

# Seleccionar solo la columna 'Close' como variable objetivo
data = df[['Datetime', 'Close']].set_index('Datetime')

# Crear un objeto MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Aplicar normalización a la columna 'Close'
data['Close'] = scaler.fit_transform(data[['Close']])

print(data.head())
print(4)
# Dividir los datos en entrenamiento (80%) y prueba (20%)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Número de pasos hacia atrás para usar como entrada
n_steps = 60  # Ejemplo: usar 60 minutos anteriores para predecir el siguiente

# Crear secuencias para entrenamiento y prueba
X_test, y_test = create_sequences(test_data['Close'], n_steps)
print(5)
X_train, y_train = create_sequences(train_data['Close'], n_steps)
print(f"Tamaño de X_train: {X_train.shape}")
print(f"Tamaño de y_train: {y_train.shape}")

# Reshaping de las secuencias para que puedan ser procesadas por la RNN (3D: [samples, timesteps, features])
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("CREANDO MODELO SECUENCIAL:")
# Crear el modelo secuencial
model = Sequential([
    Bidirectional(LSTM(100, return_sequences=False), input_shape=(n_steps, 1)),  # LSTM bidireccional
    Dropout(0.2),  # Dropout para prevenir sobreajuste
    Dense(1)  # Capa de salida
])

# Compilar el modelo con un optimizador Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Resumen del modelo
model.summary()
# Definir EarlyStopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("ENTRENANDO MODELO (DEMORA)")
# Entrenamiento del modelo
history = model.fit(
    X_train,
    y_train,
    epochs=9,          # Aumentamos las épocas
    batch_size=64,      # Tamaño de batch
    validation_split=0.1, # Para validación
    callbacks=[early_stopping], # Early stopping
    verbose=1
)



model.save('modelo_entrenado_rnn.h5')  # Guardamos el modelo completo (incluyendo pesos, arquitectura y optimizador)
print("Modelo guardado con éxito")

y_pred = model.predict(X_test)

# Guardar las predicciones en un archivo .npy
np.save('predicciones2.npy', y_pred)

print("Las predicciones se han guardado en 'predicciones.npy'.")