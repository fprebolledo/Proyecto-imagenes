import cv2
import numpy as np
from matplotlib import pyplot as plt

num = 24306

img = cv2.imread(f'images/ISIC_00{num+39}.jpg')
median = cv2.medianBlur(img,21)    
grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

ret2,th1 = cv2.threshold(grey,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,581,2)
th3 = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,581,2)
titles = ['Original Image', 'OTSU THRESHOLDING',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


