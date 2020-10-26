import cv2
import numpy as np
from matplotlib import pyplot as plt
import sklearn

num = 24306

for i in range(50):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)    
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

    ret2,binary = cv2.threshold(grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite(f"results/IMG{num+i}A.jpg",binary)
