import cv2
import numpy as np
from matplotlib import pyplot as plt

num = 24306

for i in range(50):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)
    blur = cv2.GaussianBlur(median,(7,7),0)
    
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)


    mean = np.mean(grey) * 1.2
    print(mean)
    
    rows, cols = img.shape[0], img.shape[1]
    # new_gray = np.ones((rows,cols))
    # for row in range(rows):
    #     for col in range(cols):
    #         if grey[row][col]<mean:
    #             grey[row][col] -= 12
    #         """ else:
    #             grey[row][col] += 12 """


    ret2,binary = cv2.threshold(grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # lista = [2,3,5,6,9,10,12,15,18]
    # for j in lista:
    #     #iteramos en ventanas de ixi para sacar el ruido de la imagen
    #     # se hacen iteraciones para borrar primero los ruidos pequeños
    #     # y luego los más grandes que van quedando.
    #     kernel = np.ones((j,j),np.uint8)
    #     binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    #     binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)

    print(num+i)
    cv2.imwrite(f"results/IMG{num+i}C.jpg",binary)
