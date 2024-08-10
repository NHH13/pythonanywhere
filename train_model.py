import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pickle

# Cargar los datos
data = pd.read_csv(r'data\house_prices_sinteticos.csv')  # Asegúrate de que la ruta al archivo CSV es correcta

# Seleccionar las características y la variable objetivo
X = data[['size', 'bedrooms']]
y = data['price']

# Entrenar el modelo de regresión lineal
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['price']),
                                                    data['price'],
                                                    test_size = 0.20,
                                                    random_state=42)
model.fit(X_train, y_train)

cross_val_train_MSE = cross_val_score(model,X_train,y_train, cv = 4, scoring= "neg_mean_squared_error")
cross_val_train_MAPE = cross_val_score(model,X_train,y_train, cv = 4, scoring= "neg_mean_absolute_percentage_error")
mse_cross_val = -np.mean(cross_val_train_MSE)
rmse_cross_val = np.mean([np.sqrt(-mse_fold) for mse_fold in cross_val_train_MSE])
mape_cross_val = -np.mean(cross_val_train_MAPE)
print("Train Mean Price", y_train.mean())
print("MSE Cross: ", mse_cross_val)
print("RMSE Cross: ", rmse_cross_val)
print("MAPE Cross: ", mape_cross_val)
print("**********")
print("MSE Test: ", mean_squared_error(y_test, model.predict(X_test)))
print("RMSE Test: ", np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
print("MAPE Test: ", mean_absolute_percentage_error(y_test, model.predict(X_test)) )

# Guardar el modelo entrenado en un archivo .pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modelo entrenado y guardado en 'model.pkl'")
