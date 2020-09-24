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
    image = cv2.resize(image, (500, 400))         #Convertir la imagen de 1220x1080 a 500x400
    
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
grapeWhite = io.ImageCollection('./Imagenes/Train/grapeWhite/*.png:./Imagenes/Train/grapeWhite/*.jpg')
grapeBlue = io.ImageCollection('./Imagenes/Train/grapeBlue/*.png:./Imagenes/Train/grapeBlue/*.jpg')
grapePink = io.ImageCollection('./Imagenes/Train/grapePink/*.png:./Imagenes/Train/grapePink/*.jpg')
        
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

# Analisis de grapeWhite
iter = 0
for objeto in grapeWhite:
    datos.append(Elemento())
    datos[i].pieza = 'grapeWhite'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='green', marker='o')
    i += 1
    iter += 1
print("grapeWhite OK")

# Analisis de grapeBlue
iter = 0
for objeto in grapeBlue:
    datos.append(Elemento())
    datos[i].pieza = 'grapeBlue'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='blue', marker='o')
    i += 1
    iter += 1
print("grapeBlue OK")

# Analisis de grapePink
iter = 0
for objeto in grapePink:
    datos.append(Elemento())
    datos[i].pieza = 'grapePink'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='pink', marker='o')
    i += 1
    iter += 1
print("grapePink OK")

ax.grid(True)
ax.set_title("Analisis completo de Train")

yellow_patch = mpatches.Patch(color='green', label='grapeWhite')
blue_patch = mpatches.Patch(color='blue', label='grapeBlue')
green_patch = mpatches.Patch(color='pink', label='grapePink')
plt.legend(handles=[yellow_patch, blue_patch, green_patch])

ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_zlabel('componente 4')

plt.show()

print("Analisis completo de la base de datos de Train")
print("Cantidad de imagenes analizadas: ")
print(len(datos))

