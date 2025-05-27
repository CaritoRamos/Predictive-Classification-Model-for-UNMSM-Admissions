# app.py
pip install Flask
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo y preprocesador
modelo = joblib.load('modelo_entrenado.pkl')
preprocesador = joblib.load('preprocesador.pkl')

@app.route('/')
def index():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario
    puntaje = float(request.form['puntaje'])
    postulacion = int(request.form['postulacion'])
    carrera = request.form['carrera']
    sede = request.form['sede']
    periodo = request.form['periodo']
    sexo = request.form['sexo']
    tipo_colegio = request.form['tipo_colegio']

    # Crear DataFrame para predecir
    data = pd.DataFrame([{
        'PUNTAJE': puntaje,
        'Nº POSTULACION': postulacion,
        'CARRERA': carrera,
        'SEDE': sede,
        'PERIODO': periodo,
        'SEXO': sexo,
        'TIPO_COLEGIO': tipo_colegio
    }])

    # Preprocesar y predecir
    X = preprocesador.transform(data)
    prediccion = modelo.predict(X)[0]

    resultado = '✅ Ingresó' if prediccion == 1 else '❌ No ingresó'
    return render_template('resultado.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)