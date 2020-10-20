##########################################################
## Reduccion de dimensionalidad para Hu, Haralick y HOG ##
##########################################################
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from skimage import io, color, img_as_float, img_as_ubyte, filters
import cv2

def escala_grises(image):
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gris

def normalizacion(image):
    image = cv2.resize(image, (500, 400))
    return image

def gaussian(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def sobel(image):
    image = filters.sobel(image)
    return image

#HOG: Histograma de gradientes orientados
from skimage.feature import hog

def histograma_hog(image):
    caracteristica = hog(image, block_norm='L2-Hys').ravel()
    return caracteristica

#Haralick Textura
import mahotas

def haralick(image):
    caracteristica = mahotas.features.haralick(image).mean(axis=0)
    return caracteristica

#Momentos de Hu
def hu(image):
    caracteristica = cv2.HuMoments(cv2.moments(image)).flatten()
    return caracteristica

#Determinacion de la base de datos
grapeWhite = io.ImageCollection('./Imagenes/Train/grapeWhite/*.png:./Imagenes/Train/grapeWhite/*.jpg')
grapeBlue = io.ImageCollection('./Imagenes/Train/grapeBlue/*.png:./Imagenes/Train/grapeBlue/*.jpg')
grapePink = io.ImageCollection('./Imagenes/Train/grapePink/*.png:./Imagenes/Train/grapePink/*.jpg')

grapeWhite_gray = []
grapeWhite_n = []
grapeWhite_edge = []

grapeBlue_gray = []
grapeBlue_n = []
grapeBlue_edge = []

grapePink_gray = []
grapePink_n = []
grapePink_edge = []

i = 0
for i in range(0, len(grapeWhite)):
    aux = normalizacion(grapeWhite[i])
    #aux = gaussian(aux)
    grapeWhite_n.append(aux)
    grapeWhite_gray.append(escala_grises(grapeWhite_n[i]))
    grapeWhite_edge.append(sobel(grapeWhite_gray[i]))

i = 0
for i in range(0, len(grapeBlue)):
    aux = normalizacion(grapeBlue[i])
    #aux = gaussian(aux)
    grapeBlue_n.append(aux)
    grapeBlue_gray.append(escala_grises(grapeBlue_n[i]))
    grapeBlue_edge.append(sobel(grapeBlue_gray[i]))

i = 0
for i in range(0, len(grapePink)):
    aux = normalizacion(grapePink[i])
    #aux = gaussian(aux)
    grapePink_n.append(aux)
    grapePink_gray.append(escala_grises(grapePink_n[i]))
    grapePink_edge.append(sobel(grapePink_gray[i]))
    
#Estadistica -> Aca vamos a analizar la frecuencia de aparicion para cada uva, y para eso hacemos uso de la media aritmetica y la desviacion estandar
def estadistica(array):
    
    sum = 0
    for value in array:
        sum += value
    media = sum / len(array)
    sum = 0
    for value in array:
        sum += np.power((value - media), 2)
    desviacion = np.sqrt(sum / (len(array) - 1))
    
    return media, desviacion

#HOG 
grafico_hog, ax = plt.subplots()

for objeto in grapeWhite_gray:
    grapeWhite_fhog = histograma_hog(objeto)
    media, desviacion = estadistica(grapeWhite_fhog)
    ax.plot(media, desviacion, 'o', color='yellow')
    
for objeto in grapeBlue_gray:
    grapeBlue_fhog = histograma_hog(objeto)
    media, desviacion = estadistica(grapeBlue_fhog)
    ax.plot(media, desviacion, 'o', color='blue')
    
for objeto in grapePink_gray:
    grapePink_fhog = histograma_hog(objeto)
    media, desviacion = estadistica(grapePink_fhog)
    ax.plot(media, desviacion, 'o', color='green')

ax.grid(True)
ax.set_title("Reduccion de dimensionalidad para HOG")

yellow_patch = mpatches.Patch(color='yellow', label='grapeWhite')
red_patch = mpatches.Patch(color='blue', label='grapeBlue')
blue_patch = mpatches.Patch(color='green', label='grapePink')
plt.legend(handles=[yellow_patch, red_patch, blue_patch])

plt.ylabel('Desviacion estandar')
plt.xlabel('Media aritmetica')
plt.show()

#Hu
grafico_hu, ax = plt.subplots()

for objeto in grapeWhite_edge:
    grapeWhite_fhm = hu(objeto)
    media, desviacion = estadistica(grapeWhite_fhm)
    ax.plot(media, desviacion, 'o', color='yellow')
    
for objeto in grapeBlue_edge:
    grapeBlue_fhm = hu(objeto)
    media, desviacion = estadistica(grapeBlue_fhm)
    ax.plot(media, desviacion, 'o', color='blue')
    
for objeto in grapePink_edge:
    grapePink_fhm = hu(objeto)
    media, desviacion = estadistica(grapePink_fhm)
    ax.plot(media, desviacion, 'o', color='green')

ax.grid(True)
ax.set_title("Reduccion de Dimensionalidad para Hu")

yellow_patch = mpatches.Patch(color='yellow', label='grapeWhite')
red_patch = mpatches.Patch(color='blue', label='grapeBlue')
blue_patch = mpatches.Patch(color='green', label='grapePink')
plt.legend(handles=[yellow_patch, red_patch, blue_patch])

plt.ylabel('Desviacion estandar')
plt.xlabel('Media aritmetica')
plt.show()

#Haralick
grafica_haralick, ax = plt.subplots()

for objeto in grapeWhite_gray:
    grapeWhite_fht = haralick(objeto)
    media, desviacion = estadistica(grapeWhite_fht)
    ax.plot(media, desviacion, 'o', color='yellow')
    
for objeto in grapeBlue_gray:
    grapeBlue_fht = haralick(objeto)
    media, desviacion = estadistica(grapeBlue_fht)
    ax.plot(media, desviacion, 'o', color='blue')
    
for objeto in grapePink_gray:
    grapePink_fht = haralick(objeto)
    media, desviacion = estadistica(grapePink_fht)
    ax.plot(media, desviacion, 'o', color='green')

ax.grid(True)
ax.set_title("Reduccion de Dimensionalidad para Haralick")

yellow_patch = mpatches.Patch(color='yellow', label='grapeWhite')
red_patch = mpatches.Patch(color='blue', label='grapeBlue')
blue_patch = mpatches.Patch(color='green', label='grapePink')
plt.legend(handles=[yellow_patch, red_patch, blue_patch])

plt.ylabel('Desviacion estandar')
plt.xlabel('Media aritmetica')
plt.show() 
