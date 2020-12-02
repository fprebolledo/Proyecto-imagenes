import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from adaptative import print_img, delete_min_areas

num = 24306

def segmentation_img(num, i):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,25)
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY) 

    gmm = GaussianMixture(n_components=2)
    gmm = gmm.fit(grey)

    threshold = np.mean(gmm.means_)
    binary_img = grey < threshold
    binary_img = np.uint8(binary_img)*255
    delete_min_areas(binary_img)
    ohter = 255-binary_img
    delete_min_areas(ohter)
    binary_img = 255-ohter
    cv2.imwrite(f"results/IMG{num+i}GMMG.jpg",binary_img)

if __name__=="__main__":
    for i in range(50):
        segmentation_img(num, i)