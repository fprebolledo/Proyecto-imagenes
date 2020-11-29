import cv2
import numpy as np
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

def watershed_seg(num, i, umbral=140, blur_coef=21, min_size=500):
  img    = cv2.imread('images/ISIC_00{0}.jpg'.format(num+i))
  median = cv2.medianBlur(img, blur_coef)
  grey   = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
  elevation_map = sobel(grey)
  
  markers = np.zeros_like(grey)
  markers[grey <= umbral] = 1
  markers[grey > umbral] = 2
  
  segmented = watershed(elevation_map, markers)
  
  binary = np.zeros(segmented.shape)
  binary[segmented < 2 ] = 255
  binary[segmented >= 2] = 0
  binary = binary.astype(int)
  
  binary_removed = remove_small_objects(binary, min_size=min_size)
  
  cv2.imwrite('./results/IMG{0}W.jpg'.format(num+i), binary_removed)

if __name__ == '__main__':
  num = 24306
  for i in range(50):
    watershed_seg(num, i)

        