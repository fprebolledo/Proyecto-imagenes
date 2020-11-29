import cv2
import numpy as np
from skimage.exposure import histogram
from adaptative import delete_min_areas

def histogram_seg(num, i):
  """
  Funcion que recibe el indice de la imagen
  Estudia el histograma determinando un punto
  Para determinar un umbral que nos permita segmentar de forma binaria
  """
  # lee la imagen y aplica blur bw
  img    = cv2.imread(f'images/ISIC_00{num+i}.jpg')
  median = cv2.medianBlur(img,21)    
  grey   = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

  hist, hist_centers = histogram(grey)
  auxiliar = hist[np.where(hist > np.max(hist)/4)][0]
  umbral = hist_centers[list(hist).index(auxiliar)]

  binary = np.asarray(grey).copy()
  binary[grey <= umbral] = 255
  binary[grey >= umbral] = 0

  delete_min_areas(binary)
  
  cv2.imwrite('results/IMG{0}H.jpg'.format(num+i), binary)



if __name__ == '__main__':
  num = 24306
  for i in range(50):
    histogram_seg(num, i)