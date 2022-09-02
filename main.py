import numpy as np
from skimage import io, filters
import cv2
    
def extraccion(image):
    ##PRE PROCESAMIENTO
    aux = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convertir a escala de grises
    ##FILTRACION
    aux = cv2.GaussianBlur(aux, (3, 3), 0)   #Aplicar filtro gaussiano
    aux = filters.sobel(aux)                 #Aplicar filtro Sobel o Laplaciano
    ##EXTRACCION DE RASGOS
    hu = cv2.HuMoments(cv2.moments(aux)).flatten()
    ##ANALISIS DE LAS CARACTERISTICAS -> PARA MOMENTOS DE HU
    return aux, [hu[0], hu[1], hu[3]]

#Analisis de la base de datos (Train)
CLASS_02 = io.ImageCollection('./Imagenes/Train/CLASS_02/*.png')
CLASS_03 = io.ImageCollection('./Imagenes/Train/CLASS_03/*.png')
CLASS_04 = io.ImageCollection('./Imagenes/Train/CLASS_04/*.png')
CLASS_05 = io.ImageCollection('./Imagenes/Train/CLASS_05/*.png')
CLASS_06 = io.ImageCollection('./Imagenes/Train/CLASS_06/*.png')
CLASS_07 = io.ImageCollection('./Imagenes/Train/CLASS_07/*.png')
CLASS_08 = io.ImageCollection('./Imagenes/Train/CLASS_08/*.png')
#Elemento para cada tipo de uva
class Elemento:
    def __init__(self):
        self.pieza = None
        self.image = None
        self.caracteristica = []
        self.distancia = 0
        
#Analisis de datos
datos = []
i = 0

# Analisis de CLASS_02
iter = 0
for objeto in CLASS_02:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_02'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("CLASS_02 OK")
# Analisis de CLASS_03
iter = 0
for objeto in CLASS_03:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_03'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("CLASS_03 OK")
# Analisis de CLASS_04
iter = 0
for objeto in CLASS_04:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_04'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("CLASS_04 OK")
# Analisis de CLASS_05
iter = 0
for objeto in CLASS_05:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_05'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("CLASS_05 OK")
# Analisis de CLASS_06
iter = 0
for objeto in CLASS_06:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_06'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("CLASS_06 OK")
# Analisis de CLASS_07
iter = 0
for objeto in CLASS_07:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_07'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("CLASS_07 OK")
# Analisis de CLASS_08
iter = 0
for objeto in CLASS_08:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_08'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("CLASS_08 OK")

print("Analisis completo de la base de datos de Train")
print("Cantidad de imagenes analizadas: ")
print(len(datos))

# Elemento a evaluar
test = Elemento()
numero = input("Introduce numero de la foto: ")

nombre = './Imagenes/Test/photo'+str(numero)+'.png'
image = io.imread(nombre)

test.image, test.caracteristica = extraccion(image)
test.pieza = 'CLASS_02' # label inicial 

#KNN
print("\nInicializacion KNN")
i = 0
sum = 0
for ft in datos[0].caracteristica:
        sum = sum + np.power(np.abs(test.caracteristica[i] - ft), 2)
        i += 1
d = np.sqrt(sum)

for element in datos:
    sum = 0
    i = 0
    for ft in (element.caracteristica):
        sum = sum + np.power(np.abs((test.caracteristica[i]) - ft), 2)
        i += 1
    
    element.distancia = np.sqrt(sum)
    
    if (sum < d):
        d = sum
        test.pieza = element.pieza

# Algoritmo de ordenamiento de burbuja
swap = True
while (swap):
    swap = False
    for i in range(1, len(datos)-1) :
        if (datos[i-1].distancia > datos[i].distancia):
            aux = datos[i]
            datos[i] = datos[i-1]
            datos[i-1] = aux
            swap = True
print("\nPredicciones para KNN con K=2: ")            
k = 2
for i in range(k):
    print(datos[i].pieza)

