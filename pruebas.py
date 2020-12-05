import cv2
import numpy as np
from matplotlib import pyplot as plt
from adaptative import delete_min_areas

def pruebas(num, i):
    num = 24306

    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.medianBlur(grey,31)    

    ret2,th1 = cv2.threshold(grey,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY_INV,437,2)
    th3 = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,437,2)
    # titles = ['Original Image', 'OTSU THRESHOLDING',
    #             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    # images = [grey, th1, th2, th3]

    # for i in range(4):
    #     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])
    # plt.show()
    delete_min_areas(th2)
    auxiliar = 255-th2
    delete_min_areas(auxiliar)
    binary = 255-auxiliar
    cv2.imwrite(f"results_jose/IMG{num+i}R.png", binary)

    

if __name__ == '__main__':
  num = 24306
  for i in range(50):
    pruebas(num, i)
