import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Cargar los datos
data = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Documentos\GitHub\pythonanywhere\data\house_prices_sinteticos.csv')  # Asegúrate de que la ruta al archivo CSV es correcta

# Seleccionar las características y la variable objetivo
X = data[['size', 'bedrooms']]
y = data['price']

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Guardar el modelo entrenado en un archivo .pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modelo entrenado y guardado en 'model.pkl'")
