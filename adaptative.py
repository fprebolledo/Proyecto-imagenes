import cv2
import numpy as np
from matplotlib import pyplot as plt

num = 24306

def delete_min_areas(binary_img):
    contours, _ = cv2.findContours(binary_img, 1, 2)
    # creo una lista con las áreas de los contornos
    areas = list(map(cv2.contourArea, contours))
    i = 0
    # los itero y si el area es menor que el area maxima lo eliimino
    for cont in contours:
        area = areas[i]
        if area < max(areas):
            #mascara de unos
            mask = np.zeros(binary_img.shape,np.uint8)
            cv2.drawContours(mask,[cont],0,255,-1)
            pixelpoints = np.transpose(np.nonzero(mask))
            for pint in pixelpoints:
                binary_img[pint[0]][pint[1]] = 0
        i += 1

for i in range(50):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)    
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,361,2) 
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

    # plt.imshow(binary)
    # plt.show()
    cv2.imwrite(f"results/IMG{num+i}A.jpg",binary)