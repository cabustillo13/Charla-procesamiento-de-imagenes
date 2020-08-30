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
    image = cv2.resize(image, (500, 400))         #Convertir la imagen de 1220x1080 a 500x400
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
##Entrenamiento de la base de datos 
grapeWhite = io.ImageCollection('./Imagenes/Train/grapeWhite/*.png:./Imagenes/Train/grapeWhite/*.jpg')
grapeBlue = io.ImageCollection('./Imagenes/Train/grapeBlue/*.png:./Imagenes/Train/grapeBlue/*.jpg')
grapePink = io.ImageCollection('./Imagenes/Train/grapePink/*.png:./Imagenes/Train/grapePink/*.jpg')
        
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

# Analisis de grapeWhite
iter = 0
for objeto in grapeWhite:
    datos.append(Elemento())
    datos[i].pieza = 'grapeWhite'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("grapeWhite OK")

# Analisis de grapeBlues
iter = 0
for objeto in grapeBlue:
    datos.append(Elemento())
    datos[i].pieza = 'grapeBlue'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("grapeBlues OK")

# Analisis de grapePink
iter = 0
for objeto in grapePink:
    datos.append(Elemento())
    datos[i].pieza = 'grapePink'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("grapePink OK")

print("Analisis completo de la base de datos de Train")
print("Cantidad de imagenes analizadas: ")
print(len(datos))

# Elemento a evaluar
#Recordar aplicar Transformacion.py cuando se quiera evaluar una nueva imagen.
test = Elemento()
numero = input("Introduce numero de la foto: ")

nombre = './Imagenes/Test/photo'+str(numero)+'.jpg'
image = io.imread(nombre)

test.image, test.caracteristica = extraccion(image)
test.pieza = 'grapeBlue' # label inicial 

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

# Algoritmo de ordenamiento de burbuja-> lo elegi porque es bastante estable
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

