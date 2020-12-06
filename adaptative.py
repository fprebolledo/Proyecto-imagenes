import cv2
import numpy as np
from matplotlib import pyplot as plt

num = 24306

def delete_min_areas(binary_img):
    """
    Funcion que recibe una imagen binaria
    Elimina las formas de menor area de la imagen
    """
    contours, _ = cv2.findContours(binary_img, 1, 2)
    # creo una lista con las áreas de los contornos
    areas = list(map(cv2.contourArea, contours))

    # los itero y si el area es menor que el area maxima lo eliimino
    for i, cont in enumerate(contours):
        area = areas[i]
        if area < max(areas):
            #m rellena lo que esta dentro del contorno con el valor 0
            cv2.drawContours(binary_img,[cont],0,0,-1)

def print_img(img, nombre):
    cv2.imshow(nombre, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segmentation_img(num, i):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)    
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,437,2) 
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
    auxiliar = 255-binary
    delete_min_areas(auxiliar)
    binary = 255-auxiliar

    return binary

if __name__=="__main__":
    for i in range(50):
        binary = segmentation_img(num, i)
        cv2.imwrite(f"results/IMG{num+i}A.jpg",binary)
        