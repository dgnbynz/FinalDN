import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['image.cmap'] = 'gray'
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, color, img_as_float, filters
from skimage.feature import hog
import cv2
import mahotas
    
def extraccion(image):
    
    ##TRANSFORMACION
    #Recordar hacer la transformacion de la imagen con el programa Transformacion.py
    image = cv2.resize(image, (384, 216))         #Convertir la imagen de 1220x1080 a 500x400
    
    ##PRE PROCESAMIENTO
    aux = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convertir a escala de grises
    
    ##FILTRACION
    aux = cv2.GaussianBlur(aux, (3, 3), 0)   #Aplicar filtro gaussiano
    aux = filters.sobel(aux)                 #Aplicar filtro Sobel o Laplaciano
            
    ##EXTRACCION DE RASGOS
    #haralick=mahotas.features.haralick(aux).mean(axis=0)
    hu = cv2.HuMoments(cv2.moments(aux)).flatten()
    
    ##ANALISIS DE LAS CARACTERISTICAS
    #PARA MOMENTOS DE HU
    return aux, [hu[0], hu[1], hu[3]]

#Analisis de la base de datos (YTrain)
##Entrenamiento de la base de datos 
CLASS_02 = io.ImageCollection('./Imagenes/Train/CLASS_02/*.png')
CLASS_03 = io.ImageCollection('./Imagenes/Train/CLASS_03/*.png')
CLASS_04 = io.ImageCollection('./Imagenes/Train/CLASS_04/*.png')
CLASS_05 = io.ImageCollection('./Imagenes/Train/CLASS_05/*.png')
CLASS_06 = io.ImageCollection('./Imagenes/Train/CLASS_06/*.png')
CLASS_07 = io.ImageCollection('./Imagenes/Train/CLASS_07/*.png')
CLASS_08 = io.ImageCollection('./Imagenes/Train/CLASS_08/*.png')
        
#Elemento de ferreteria
class Elemento:
    def __init__(self):
        self.pieza = None
        self.image = None
        self.caracteristica = []
        self.distancia = 0
        
#Analisis de datos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

datos = []
i = 0

# Analisis de CLASS_02
iter = 0
for objeto in CLASS_02:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_02'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='yellow', marker='o')
    i += 1
    iter += 1
print("CLASS_02 OK")

# Analisis de CLASS_03
iter = 0
for objeto in CLASS_03:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_03'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='blue', marker='o')
    i += 1
    iter += 1
print("CLASS_03 OK")

# Analisis de CLASS_04
iter = 0
for objeto in CLASS_04:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_04'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='green', marker='o')
    i += 1
    iter += 1
print("CLASS_04 OK")

# Analisis de CLASS_05
iter = 0
for objeto in CLASS_05:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_05'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='black', marker='o')
    i += 1
    iter += 1
print("CLASS_05 OK")

# Analisis de CLASS_06
iter = 0
for objeto in CLASS_06:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_06'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='red', marker='o')
    i += 1
    iter += 1
print("CLASS_06 OK")

# Analisis de CLASS_07
iter = 0
for objeto in CLASS_07:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_07'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='pink', marker='o')
    i += 1
    iter += 1
print("CLASS_07 OK")

# Analisis de CLASS_08
iter = 0
for objeto in CLASS_08:
    datos.append(Elemento())
    datos[i].pieza = 'CLASS_08'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='purple', marker='o')
    i += 1
    iter += 1
print("CLASS_08 OK")


ax.grid(True)
ax.set_title("Analisis completo de Train")

yellow_patch = mpatches.Patch(color='yellow', label='CLASS_02')
blue_patch = mpatches.Patch(color='blue', label='CLASS_03')
green_patch = mpatches.Patch(color='green', label='CLASS_04')
black_patch = mpatches.Patch(color='black', label='CLASS_05')
red_patch = mpatches.Patch(color='red', label='CLASS_06')
pink_patch = mpatches.Patch(color='pink', label='CLASS_07')
purple_patch = mpatches.Patch(color='purple', label='CLASS_08')


plt.legend(handles=[yellow_patch, blue_patch, green_patch, black_patch, red_patch, pink_patch, purple_patch])

ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_zlabel('componente 4')

plt.show()

print("Analisis completo de la base de datos de Train")
print("Cantidad de imagenes analizadas: ")
print(len(datos))

