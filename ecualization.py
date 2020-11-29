import cv2
import numpy as np
from matplotlib import pyplot as plt
from adaptative import delete_min_areas

num = 24306

def segmentation_img(num, i):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)    
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey) 
    ## gausiano
    #binary = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY_INV,391,2)
    # con otsu
    _, binary = cv2.threshold(grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    lista = [2,3,5]
    for j in lista:
        #iteramos en ventanas de ixi para sacar el ruido de la imagen
        # se hacen iteraciones para borrar primero los ruidos pequeños
        # y luego los más grandes que van quedando.
        kernel = np.ones((j,j),np.uint8)
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
        # quiza no hacer estos tan grandes, si no que hacerlos chicos, hasta 6 y luego cerrar las ciurvas que se puedan.
    

    delete_min_areas(binary)
    ohter = 255-binary
    delete_min_areas(ohter)
    binary = 255-ohter
    cv2.imwrite(f"results/IMG{num+i}EQO.jpg",binary)


if __name__=="__main__":
    for i in range(50):
        segmentation_img(num, i)