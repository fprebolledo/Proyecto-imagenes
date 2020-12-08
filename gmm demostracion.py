import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from adaptative import print_img, delete_min_areas, segmentation_img

num = 24306

def segmentation_img_gmm(path, num, i, a=False):
    img = cv2.imread(path)
    cv2.imshow("Imagen original", img)
    cv2.waitKey(0)
    median = cv2.medianBlur(img,21)  
    cv2.imshow("Medianblur de 21x21", median)
    cv2.waitKey(0)

    grey =   cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    _, s, v = cv2.split(cv2.cvtColor(median, cv2.COLOR_BGR2HSV))
    cv2.imshow("Canal s de HSV", s)
    cv2.waitKey(0)
    ## metodo de gausianas multiples de sklearn
    gmm = GaussianMixture(n_components=2)
    gmm = gmm.fit(s)

    threshold = np.mean(gmm.means_)
    #si utilizamos el canal v el > se debe dar vuelta al otro lado

    binary_img = s > threshold
    binary_img = np.uint8(binary_img)*255
    cv2.imshow("Imagen binarizada por threshold de gaussian mixture", binary_img)
    cv2.waitKey(0)
    # quitamos areas pequeñas
    delete_min_areas(binary_img)
    ohter = 255-binary_img
    delete_min_areas(ohter)
    binary_img = 255-ohter
    cv2.imshow("Areas pequenas (negras y blancas) eliminadas (gaussian mixture)", binary_img)
    cv2.waitKey(0)
    if a:
        std_grey = np.std(grey)
        avg_grey = np.average(grey)
        adaptive_seg = segmentation_img(num, i)
        if std_grey<13.5 and avg_grey<170:
            binary_img = binary_img
            cv2.imshow("Resultado", binary_img)
            cv2.waitKey(0)

        else:
            cv2.imshow("Resultado de threshold adaptativo", adaptive_seg)
            cv2.waitKey(0)

            binary_img = np.logical_and(binary_img.flatten(), adaptive_seg.flatten())
            binary_img = binary_img.reshape(s.shape)
            binary_img = np.uint8(binary_img)*255
            cv2.imshow("Logical and entre threshold adaptativo y gaussian mixture", binary_img)
            cv2.waitKey(0)
    return binary_img

    
if __name__=="__main__":
    a = True
    for i in range(50):
        path = f'images/ISIC_00{num+i}.jpg'
        result = segmentation_img_gmm(path, num, i, a, False)
        seg_real = cv2.imread(f"images/ISIC_00{num+i}_segmentation.png")
        img_real = cv2.imread(path)
        cv2.destroyAllWindows()
        cv2.imshow("imagen original", img_real)
        cv2.imshow("resultado", result)
        cv2.imshow("segmentacion real", seg_real)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if a:
            cv2.imwrite(f"results/IMG{num+i}GMMSA.jpg", result)
        else:
            cv2.imwrite(f"results/IMG{num+i}GMMS.jpg", result)
    # Para correr imagenes del profe
    # for i in range(5):
    #     path = f'profe/ISIC_00{i}.jpg'
    #     segmentation_img_gmm(path, 0, 0, True, False)
