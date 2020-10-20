#Imagen a analizar
#Las fotos de entrada estan en formato png o jpeg
prueba = './Imagenes/Test/photo8.jpg'

#####################################################################################################################################
##Filtro Gaussiano
#####################################################################################################################################
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import numpy as np
import cv2
from skimage import io,filters

tornillo = io.imread(prueba)
#El modulo io tiene utilidades para leer y escribir imagenes en varios formatos.
#io.imread lectura y escritura de las imagenes via imread

#Redimensionamiento de la imagen de entrada
fixed_size = tuple((500, 400))
tornillo = cv2.resize(tornillo, fixed_size) 

#Gauss
bs0 = filters.gaussian(tornillo, sigma=1)
bs1 = filters.gaussian(tornillo, sigma=3)
bs2 = filters.gaussian(tornillo, sigma=5)
bs3 = filters.gaussian(tornillo, sigma=15)

f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(16, 5))
ax0.imshow(bs0)
ax0.set_title('$\sigma=1$')
ax1.imshow(bs1)
ax1.set_title('$\sigma=3$')
ax2.imshow(bs2)
ax2.set_title('$\sigma=5$')
ax3.imshow(bs2)
ax3.set_title('$\sigma=15$')
plt.show()

#Bibliografia consultada
#https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian

#####################################################################################################################################
##Filtro Sobel y Roberts
#####################################################################################################################################
from skimage import color

image = io.imread(prueba)
image = color.rgb2gray(image)

edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)

sobel_v=filters.sobel_v(image)
sobel_h=filters.sobel_h(image)

fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Operador cruzado de Robert')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Operador de Sobel')

axes[2].imshow(sobel_v, cmap=plt.cm.gray)
axes[2].set_title('Operador de Sobel vertical')

axes[3].imshow(sobel_h, cmap=plt.cm.gray)
axes[3].set_title('Operador de Sobel horizontal')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

#Bibliografia consultada
#https://scikit-image.org/docs/dev/auto_examples/edges/plot_edge_filter.html#sphx-glr-auto-examples-edges-plot-edge-filter-py

#####################################################################################################################################
##Filtro Gaussiano + Sobel
#####################################################################################################################################
from skimage import img_as_float, img_as_ubyte

image = io.imread(prueba)
tornillo = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
tornillo = cv2.resize(tornillo, (500,400))

bg = cv2.GaussianBlur(tornillo, (3, 3), 0)
bc = filters.sobel(tornillo)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 15))
ax0.imshow(tornillo)
ax0.set_title('Original')
ax1.imshow(bg)
ax1.set_title('Filtro Gauss')
ax2.imshow(-bc)
ax2.set_title("Filtros Gauss+Sobel")
plt.show()

#####################################################################################################################################
##Perona Malik
#####################################################################################################################################
from scipy import misc, ndimage

#Inicializacion
iterations = 30
delta = 0.14
kappa = 15

#Convertir la imagen de entrada
im = misc.imread(prueba, flatten=True)
im = im.astype('float64')

#Condicion inicial
u = im

# Distancia al pixel central
dx = 1
dy = 1
dd = np.sqrt(2)

#2D diferentes finitas ventanas
windows = [
    np.array(
            [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
    ),
    np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
    ),
]

for r in range(iterations):
    #Aproximacion de gradientes
    nabla = [ ndimage.filters.convolve(u, w) for w in windows ]

    #Aproximacion de la funcion de difusion
    diff = [ 1./(1 + (n/kappa)**2) for n in nabla]

    #Actualizar imagen
    terms = [diff[i]*nabla[i] for i in range(4)]
    terms += [(1/(dd**2))*diff[i]*nabla[i] for i in range(4, 8)]
    u = u + delta*(sum(terms))


# Kernel para el gradiente en la direccion x
Kx = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
)
# Kernel para el gradiente en la direccion y
Ky = np.array(
    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
)
#Aplicar kernel a la imagen
Ix = ndimage.filters.convolve(u, Kx)
Iy = ndimage.filters.convolve(u, Ky)

#Retorna (Ix, Iy)
G = np.hypot(Ix, Iy)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 15))
ax0.imshow(im)
ax0.set_title('Original')
ax1.imshow(u)
ax1.set_title('Despues de la difusion')
ax2.imshow(G)
ax2.set_title("Gradiente despues de la difusion")
plt.show()

#Bibliografia consultada
#https://github.com/fubel/PeronaMalikDiffusion/blob/master/main.py

#####################################################################################################################################
##Filtro Laplace, Median, Frangi y Prewitt
#####################################################################################################################################
#GAUSS
image = io.imread(prueba)
gris = color.rgb2gray(image)

#OTROS FILTROS
laplace = filters.laplace(gris)
median = filters.median(gris)
frangi = filters.frangi(gris)
prewitt = filters.prewitt(gris)
 
f, (ax0, ax1,ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(16, 5))
ax0.imshow(image)
ax0.set_title('Original')

ax1.imshow(frangi)
ax1.set_title('Filtro Frangi')

ax2.imshow(prewitt)
ax2.set_title('Filtro Prewitt')

ax3.imshow(laplace)
ax3.set_title('Filtro Laplace')

ax4.imshow(median)
ax4.set_title('Filtro Median')
plt.show()

#Bibliografia consultada
#https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sobel 
