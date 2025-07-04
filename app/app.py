from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Cargar modelo entrenado
modelo = joblib.load('modelo_entrenado.pkl')

# Leer carreras desde archivo txt
with open('carreras.txt', encoding='utf-8') as f:
    carreras = sorted([line.strip().upper() for line in f if line.strip()])

@app.route('/')
def index():
    return render_template('formulario.html', carreras=carreras)

@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener datos del formulario
    carrera = request.form['CARRERA'].upper()

    datos = {
        'CARRERA': [carrera],
        'PUNTAJE': [int(request.form['PUNTAJE'])],
        'SEDE': [request.form['SEDE']],
        'PERIODO': [request.form['PERIODO']],
        'Nº POSTULACION': [int(request.form['Nº POSTULACION'])],
        'SEXO': [request.form['SEXO']],
        'TIPO_COLEGIO': [request.form['TIPO_COLEGIO']],
    }

    df = pd.DataFrame(datos)

    # Hacer predicción
    pred = modelo.predict(df)[0]

    resultado = "INGRESAS" if pred == 1 else "NO INGRESAS"

    return render_template("resultado.html", resultado=resultado)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
