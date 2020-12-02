import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from adaptative import print_img, delete_min_areas

num = 24306

def segmentation_img(num, i):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,25)    
    _, s, v = cv2.split(cv2.cvtColor(median, cv2.COLOR_BGR2HSV))
   
    ## metodo de gausianas multiples de sklearn
    gmm = GaussianMixture(n_components=2)
    gmm = gmm.fit(s)

    threshold = np.mean(gmm.means_)
    #si utilizamos el canal v el > se debe dar vuelta al otro lado

    binary_img = s > threshold
    binary_img = np.uint8(binary_img)*255

    # quitamos areas pequeñas
    delete_min_areas(binary_img)
    ohter = 255-binary_img
    delete_min_areas(ohter)
    binary_img = 255-ohter

    #prints para ver si se puede mejorar
    # cv2.imshow('s',s)
    # cv2.imshow("v", v)
    # cv2.imshow("binary", binary_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f"results/IMG{num+i}GMMS.jpg",binary_img)

if __name__=="__main__":
    for i in range(50):
        segmentation_img(num, i)
