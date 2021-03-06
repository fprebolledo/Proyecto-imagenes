import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from adaptative import print_img, delete_min_areas, segmentation_img
num = 24306

def segmentation_img_gmm(num, i, a=False):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)  
    grey =   cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
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

    if a:
        std_grey = np.std(grey)
        avg_grey = np.average(grey)
        adaptive_seg = segmentation_img(num, i)

        if std_grey<13.5 and avg_grey<170:
            binary_img = binary_img
        else:
            binary_img = np.logical_and(binary_img.flatten(), adaptive_seg.flatten())
            binary_img = binary_img.reshape(s.shape)
            binary_img = np.uint8(binary_img)*255
    
    if a:
        cv2.imwrite(f"results/IMG{num+i}GMMSA.jpg",binary_img)
    else:
        cv2.imwrite(f"results/IMG{num+i}GMMS.jpg",binary_img)

if __name__=="__main__":
    for i in range(50):
        segmentation_img_gmm(num, i, False)
