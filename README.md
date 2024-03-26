# NLP_study
Clasificador de textos

Para cargar y utilizar el clasificador de textos:

1. Descarga el archivo 'clasificador_texto.pkl'.
2. En tu código de Python, carga el clasificador utilizando la biblioteca joblib:
   ```python
   import joblib

   # Cargar el clasificador desde el archivo
   clasificador = joblib.load('clasificador_texto.pkl')

#Utiliza el clasificador
oracion_prueba = 'Texto que deseas clasificar'
resultado = clasificador.classify(oracion_prueba)
print("Resultado de la clasificación:", resultado)
