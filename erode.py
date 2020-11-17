import cv2
import numpy as np
from matplotlib import pyplot as plt

num = 24306

for i in range(50):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)    
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,581,2)
    lista = [2,3,5,6,9]
    for j in lista:
        #iteramos en ventanas de ixi para sacar el ruido de la imagen
        # se hacen iteraciones para borrar primero los ruidos pequeños
        # y luego los más grandes que van quedando.
        kernel = np.ones((j,j),np.uint8)
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
        binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
        # quiza no hacer estos tan grandes, si no que hacerlos chicos, hasta 6 y luego cerrar las ciurvas que se puedan.
   
    contours,hierarchy = cv2.findContours(binary, 1, 2)
    plt.imshow(binary)
    plt.show()
    prom = 0
    areas = list(map(cv2.contourArea, contours))
    other= 255-binary
    
    for cont in contours:
        area = cv2.contourArea(cont)
        prom += area
        if area < max(areas):
            mask = np.zeros(grey.shape,np.uint8)
            cv2.drawContours(mask,[cont],0,255,-1)
            pixelpoints = np.transpose(np.nonzero(mask))
            for pint in pixelpoints:
                binary[pint[0]][pint[1]] = 0

    print(prom/len(contours))
    plt.imshow(binary)
    plt.show()
    break
    cv2.imwrite(f"results/IMG{num+i}ME.jpg",binary)
    #generalizar el área 