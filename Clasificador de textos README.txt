Este es un proyecto de clasificación de textos que utiliza el algoritmo de clasificación Naive Bayes implementado con la biblioteca NLTK en Python. El clasificador puede etiquetar automáticamente las oraciones como positivas o negativas según su contenido.

El archivo clasificador_textos cuenta con dos frases de prueba (neg/pos), mientras que clasificador_texto_libre deja vía libre para clasificar textos. Este código es el que se ve más adelante para cargar el clasificador utilizando la biblioteca joblib.

# Instrucciones de uso
# Requisitos previos
Asegúrate de tener instaladas las siguientes bibliotecas de Python:
- NLTK (Natural Language Toolkit)
- joblib

Puedes instalar NLTK ejecutando el siguiente comando:

pip install nltk

Y joblib con:

pip install joblib

Además, necesitarás descargar los datos adicionales de NLTK ejecutando el siguiente script de Python:

import nltk
nltk.download('punkt')

# Cómo utilizar el clasificador
1. Descarga el archivo 'clasificador_texto.pkl' desde este repositorio.
2. Coloca el archivo 'clasificador_texto.pkl' en el mismo directorio que tu script de Python desde el cual deseas cargarlo.
3. En tu script de Python, carga el clasificador utilizando la biblioteca joblib:

import joblib
from nltk.tokenize import word_tokenize

# Cargar el clasificador desde el archivo
clasificador = joblib.load('clasificador_texto_libre.pkl')

# Tokenizar la oración de prueba
oracion_prueba = 'Texto que deseas clasificar'
palabras_oracion = word_tokenize(oracion_prueba.lower())

# Crear el conjunto de características
caracteristicas_oracion = {palabra: True for palabra in palabras_oracion}

# Utilizar el clasificador
resultado = clasificador.classify(caracteristicas_oracion)
print("Resultado de la clasificación:", resultado)