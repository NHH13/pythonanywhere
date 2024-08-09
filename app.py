from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import pandas as pd
import os
import json
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
from flask import Response



app = Flask(__name__)

root_path = '/home/franpujalte/pythonanywhere/'
# Directorio donde se encuentran los archivos CSV
DATA_DIR = 'data/'

# Función para cargar el modelo
def load_model():
    with open(root_path+'model.pkl', 'rb') as file:
        return pickle.load(file)

# Cargar el modelo entrenado
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')

@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    return render_template('retrain.html')

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/api/v1/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    size = float(data['size'])  # Convertir a float para asegurar que sean numéricos
    bedrooms = float(data['bedrooms'])  # Convertir a float para asegurar que sean numéricos

    # Realizar la predicción
    prediction = model.predict([[size, bedrooms]])

    return jsonify({'predictions': prediction[0]})

@app.route('/api/v1/retrain', methods=['POST'])
def retrain_api():
    global model
    # Cargar los datos desde el nuevo archivo CSV para reentrenamiento
    data = pd.read_csv(root_path+os.path.join(DATA_DIR, 'house_prices_retrain_sinteticos.csv'))  # Asegúrate de que la ruta al archivo CSV es correcta

    # Convertir las columnas a tipo numérico
    data['size'] = pd.to_numeric(data['size'], errors='coerce')
    data['bedrooms'] = pd.to_numeric(data['bedrooms'], errors='coerce')
    data['price'] = pd.to_numeric(data['price'], errors='coerce')

    # Eliminar filas con valores nulos (si hay)
    data = data.dropna()

    # Seleccionar las características y la variable objetivo
    X = data[['size', 'bedrooms']]
    y = data['price']

    # Reentrenar el modelo de regresión lineal
    new_model = LinearRegression()
    new_model.fit(X, y)

    # Guardar el modelo reentrenado en un archivo .pkl
    with open(root_path+'model.pkl', 'wb') as file:
        pickle.dump(new_model, file)

    # Actualizar el modelo en memoria
    model = load_model()

    return jsonify({'message': 'Modelo reentrenado con éxito'})

@app.route('/api/v1/visualize', methods=['GET'])
def visualize_api():
    file_name = request.args.get('file')
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.isfile(root_path+file_path):
        return jsonify({'error': 'Archivo no encontrado'}), 404

    try:
        df = pd.read_csv(root_path+file_path)
        df = df.rename(columns={'size': 'Size (m^2)', 'bedrooms': 'Bedrooms', 'price':'Price (€)'})
        data = [OrderedDict(row) for _, row in df.iterrows()]
        json_data = json.dumps(data, ensure_ascii=False)

        #data = df.to_dict(orient='records')
        #return jsonify(data)
        return Response(json_data, mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/files', methods=['GET'])
def list_files():
    try:
        # Listar todos los archivos CSV en el directorio de datos
        files = [f for f in os.listdir(root_path+DATA_DIR) if f.endswith('.csv')]
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
