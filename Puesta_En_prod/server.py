import joblib
import numpy as np

from flask import Flask
from flask import jsonify # Herramienta para trabajar con archivos json

app = Flask(__name__) # Que me cree la variable app con el nombre de este archivo

#POSTMAN PARA PRUEBAS
# Wraper en donde se especifica la ruta para contestar.
@app.route('/predict', methods=['GET']) # Basicamente deme algo
def predict():
    X_test = np.array([7.594444821,
                       7.479555538,
                       1.616463184,
                       1.53352356,
                       0.796666503,
                       0.635422587,
                       0.362012237,
                       0.315963835,
                       2.277026653])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'prediccion' : list(prediction)}) # Pasar de un diccionario de python a un json

if __name__ == "__main__":
    model = joblib.load('./models/best_model.pkl') # Cargar el modelo que vamos a usar
    app.run(port=8080) # Correr la app en el puerto especificado

    # Para la prueba se corre en el terminal $python server.py y luego se va al navegador
    # en donde se agrega la url localhost:8080/predict y debe mostrar en pantalla un archivo
    # json con la prediccion.