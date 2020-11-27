import cv2
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from adaptative import delete_min_areas

num = 24306

for i in range(50):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)    
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

    ret2,binary = cv2.threshold(grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    delete_min_areas(binary)
    ohter = 255-binary
    delete_min_areas(ohter)
    binary = 255-ohter
    
    cv2.imwrite(f"results/IMG{num+i}O.jpg",binary)
