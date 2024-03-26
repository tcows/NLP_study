from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from itertools import chain
import joblib

#Esto es un array con la oracion y la clasificacion.
#Se puede cargar así o directamente desde un archivo.

oraciones_entrenamiento = [
('El cielo está despejado y el sol brilla en el horizonte.', 'pos'),
('Los pájaros cantan alegremente en los árboles del jardín.', 'pos'),
('El río fluye tranquilamente entre las rocas y la vegetación.','pos'),
('Las estrellas brillan en el cielo nocturno y la Luna ilumina el camino.','pos'),
('La lluvia cae suavemente sobre el tejado de la casa.', 'neg'),
('El viento sopla con fuerza y las hojas de los árboles se agitan.', 'neg'),
('La nieve cubre el paisaje y todo parece estar en silencio.', 'neg'),
('El mar está agitado y las olas golpean la costa con fuerza.', 'neg'),
('Los rayos del sol iluminan el campo, creando un paisaje resplandeciente.', 'pos'),
('El canto de los pájaros llena el aire con melodías alegres y reconfortantes.', 'pos'),
('Las flores despliegan sus pétalos de colores vibrantes, embelleciendo el entorno.', 'pos'),
('El aroma de la primavera impregna el ambiente, despertando los sentidos con su dulzura.', 'pos'),
('Las montañas se yerguen majestuosas, ofreciendo vistas impresionantes y llenas de serenidad.', 'pos'),
('El río serpentea entre los árboles, creando un espectáculo natural de paz y armonía.', 'pos'),
('El brillo de las estrellas en la noche crea un telón de fondo mágico y relajante.', 'pos'),
('El suave murmullo del arroyo tranquiliza el alma, invitando a la contemplación y la calma.', 'pos'),
('El viento acaricia suavemente la piel, trayendo consigo una sensación de libertad y frescura.', 'pos'),
('El aroma de la lluvia recién caída impregna el aire, renovando la tierra y revitalizando los sentidos.', 'pos'),
('La oscuridad de la noche envuelve el paisaje, creando una sensación de desasosiego y temor.', 'neg'),
('El frío viento del invierno corta la piel, haciendo que cada paso sea una lucha contra el gélido clima.', 'neg'),
('La lluvia cae sin cesar, empañando la vista y sumiendo el ánimo en la melancolía.', 'neg'),
('El sonido de los truenos retumba en el cielo, anunciando una tormenta inminente y perturbadora.', 'neg'),
('El cielo está cubierto de nubes grises, presagiando una jornada sombría y poco alentadora.', 'neg'),
('El silencio sepulcral del lugar inspira una sensación de inquietud y desconcierto.', 'neg'),
('La bruma densa envuelve el entorno, dificultando la visión y generando una sensación de claustrofobia.', 'neg'),
('El aire está cargado de humedad, pesando sobre los hombros y dificultando la respiración.', 'neg'),
('Los árboles se agitan con violencia, cediendo ante la fuerza desatada de la tormenta.', 'neg'),
('El paisaje está cubierto de niebla, ocultando cualquier atisbo de belleza y sumiendo todo en la penumbra.', 'neg')
]

#Se separan todas las palabras en una lista única (chain = optimizador de ciclos)
#Para el clasificador puede ser un problema que las oraciones tengan diferente número de palabras
#Se hace una sola instancia de cada palabra para que no se repitan
#Es más eficiente que usar for para grandes volúmenes de datos

vocabulario = set(chain(*[word_tokenize(i[0].lower()) for i in oraciones_entrenamiento]))

#Se genera para cada oración si existe o no la palabra de la lista total
#Esta es la matriz dimensional

array_palabras = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulario}, tag) for sentence, tag in oraciones_entrenamiento]
for i in array_palabras:
    print ('Oracion: ')
    print (i)
    print('_____________________________________________________________________________')  
    
#Ahora se puede ingresar el conjunto de datos en el modelo
#Este es el primer clasificador
#Le decimos que se entrene

NaiveBClassifier = NaiveBayesClassifier.train(array_palabras)

# Solicitar al usuario que ingrese una oración para clasificar
oracion = input("Ingrese una oración para clasificar: ")

# Preprocesar la oración de entrada
tokens_oracion = word_tokenize(oracion.lower())
array_oracion = {word: True for word in vocabulario if word in tokens_oracion}

# Clasificar la oración usando el modelo entrenado
clasificacion = NaiveBClassifier.classify(array_oracion)

# Imprimir el resultado de la clasificación
print("Resultado de la clasificación para la oración ingresada:", clasificacion)

nombreFile = './models/NaiveBClassifier.joblib.pkl'
rtn = joblib.dump(NaiveBClassifier, nombreFile, compress=9)