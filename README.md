# Despliegue de un Modelo de Regresión Lineal en PythonAnywhere
![Python Anywhere](https://media.licdn.com/dms/image/C561BAQHDnw3jPc3HsA/company-background_10000/0/1588183934551/pythonanywhere_cover?e=2147483647&v=beta&t=BTrp5lz4RhgRYVIWpbpPpQKnf7ULDxbYM57GV47pA_A)

## Descripción del Proyecto

Este proyecto consiste en desplegar un modelo de Machine Learning de regresión lineal que predice precios de casas basándose en características como el tamaño y el número de habitaciones. El modelo está disponible a través de una API REST accesible públicamente, permitiendo que cualquier usuario realice predicciones mediante solicitudes HTTP.

### Objetivos

- **Despliegue del Modelo**: Hacer el modelo accesible mediante una API REST a través de un entorno de hosting público, como PythonAnywhere.
- **Predicciones a través de Endpoints**: Proveer un endpoint para realizar predicciones en función de los datos proporcionados por el usuario.
- **Landing Page Informativa**: Crear una página de inicio que informe sobre cómo acceder a los endpoints disponibles y utilizar la API.
- **Redespliegue Sencillo**: Demostrar la capacidad de modificar y redesplegar la aplicación sin cambiar código en el servidor, más allá de la configuración web.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

TEAM_CHALLENGE_PYTHONANYWHERE/
├── data/
│   ├── house_prices_retrain_sinteticos.csv  # Dataset de entrenamiento para el reentrenamiento del modelo
│   └── house_prices.csv                     # Dataset original para visualización y predicción
├── static/
│   ├── css/
│   │   └── styles.css                       # Archivo de estilos CSS para el diseño de la aplicación
│   └── images/
│       ├── prediccion.jpg                   # Imagen para la sección de predicción
│       ├── retrain.jpg                     # Imagen para la sección de reentrenamiento
│       └── visualizacion.jpg               # Imagen para la sección de visualización
├── templates/
│   ├── index.html                           # Página principal con información sobre la API y enlaces a otras funciones
│   ├── predict.html                         # Página para ingresar datos y obtener predicciones
│   ├── retrain.html                         # Página para iniciar el reentrenamiento del modelo
│   └── visualize.html                       # Página para visualizar datasets en forma de tablas
├── model.pkl                                # Archivo de modelo serializado (pickle) para predicciones
├── app.py                                   # Script principal de la aplicación Flask
└── README.txt                                # Documento que describe el proyecto y su estructura

## Instrucciones de Uso

1. **Inicio de la Aplicación**
   - Ejecuta el script `app.py` para iniciar el servidor Flask.

2. **Acceso a la Aplicación**
   - Navega a `http://localhost:5000/` para acceder a la página principal.

3. **Funcionalidades**
   - **Predicción**: Accede a `/predict` para introducir datos y obtener una predicción del precio de una casa.
   - **Reentrenamiento**: Accede a `/retrain` para reentrenar el modelo con nuevos datos.
   - **Visualización**: Accede a `/visualize` para ver los datos de los archivos CSV en formato de tabla.

4. **Endpoints de la API**
   - **Predicción**: Envía una solicitud POST a `/api/v1/predict` con `size` y `bedrooms` en el cuerpo de la solicitud para obtener una predicción en euros.
   - **Reentrenamiento**: Envía una solicitud POST a `/api/v1/retrain` para reentrenar el modelo con el archivo CSV más reciente.
   - **Visualización**: Envía una solicitud GET a `/api/v1/visualize?file=nombre_del_archivo` para obtener los datos del archivo especificado en formato JSON.

## Configuración

Asegúrate de tener instaladas las dependencias necesarias. Puedes instalar las librerías requeridas con:

pip install -r requirements.txt

## Contribuciones

Las contribuciones al proyecto son bienvenidas. Si encuentras errores o tienes sugerencias para mejorar la aplicación, por favor abre un issue o envía una solicitud de extracción.

## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).
