import cv2
import numpy as np
from adaptative import delete_min_areas
from skimage.segmentation import random_walker

# https://scikit-image.org/docs/0.12.x/auto_examples/segmentation/plot_random_walker_segmentation.html#id2

num = 24306

def seg_random_walk(umbral=85, k=3):
  """
  Funcion que utiliza un umbral y una cantidad de fases
  para encontrar los marcadores de los puntos destacados de la
  imagen y genera una segmentación con una imagen binaria
  """
  for i in range(2):
    # lectura de la imagen y blur para los pelos
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    img = cv2.medianBlur(img, 21)
    
    # Pasamos a blanco y negro y usamos el negativo
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (255-img)

    # matriz que almacenará las fases para el algoritmo
    b = np.zeros_like(img)

    # se ocupan 3 fases en torno a nuestro umbral
    for j in range(k):
      b[img > umbral+5*j] = j+1

    # Aplicamos el algoritmo random walker
    labels = random_walker(img, b, beta=10)

    # obtenemos una imagen binaria
    binary = np.zeros_like(labels)
    binary[labels > 2] = 255

    # eliminamos las areas pequeñas
    delete_min_areas(binary)

    # guardamos la imagen
    cv2.imwrite('results/IMG{0}RW-{1}.jpg'.format(num+i, umbral), binary)