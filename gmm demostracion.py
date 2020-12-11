import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from adaptative import delete_min_areas, segmentation_img

num = 24306

def print_img(name, img, allow):
    if allow:
        cv2.imshow(name, img)
        cv2.waitKey(0)
    
def segmentation_img_gmm(path, a=False, prints=False):
    img = cv2.imread(path)
    print_img(f"Imagen original {i}", img, prints)
    median = cv2.medianBlur(img,21)  
    print_img("Medianblur de 21x21", median, prints)


    grey =   cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    _, s, v = cv2.split(cv2.cvtColor(median, cv2.COLOR_BGR2HSV))
    print_img("Canal s de HSV", s, prints)

    ## metodo de gausianas multiples de sklearn
    gmm = GaussianMixture(n_components=2)
    gmm = gmm.fit(s)

    threshold = np.mean(gmm.means_)
    #si utilizamos el canal v el > se debe dar vuelta al otro lado

    binary_img = s > threshold
    binary_img = np.uint8(binary_img)*255
    print_img("Imagen binarizada por threshold de gaussian mixture", binary_img, prints)

    # quitamos areas pequeñas blancas
    ohter = 255-binary_img
    delete_min_areas(ohter)
    # quitamos areas pequeñas negras
    binary_img = 255-ohter
    delete_min_areas(binary_img)
    print_img("Areas pequenas (negras y blancas) eliminadas (gaussian mixture)", binary_img, prints)

    if a:
        std_grey = np.std(grey)
        avg_grey = np.average(grey)
        adaptive_seg = segmentation_img(path)
        if std_grey<13.5 and avg_grey<170:
            ## imagenes muy claras funcionan mal con adaptativo.
            binary_img = binary_img
            print_img("Resultado", binary_img, prints)
        else:
            print_img("Resultado de threshold adaptativo", adaptive_seg, prints)
            binary_img = np.logical_and(binary_img.flatten(), adaptive_seg.flatten())
            binary_img = binary_img.reshape(s.shape)
            binary_img = np.uint8(binary_img)*255
            print_img("Logical and entre threshold adaptativo y gaussian mixture", binary_img, prints)
        
    return binary_img

def print_results(real_segmentation, obt_segmentation ):
    _, obtenido = cv2.threshold(obt_segmentation, 127, 255, cv2.THRESH_BINARY)
    true_labels, pred_labels = real_segmentation.flatten(), obtenido.flatten()
    ## todas estas funciones devuelven un array de trues y falses si el valor coincide y luego se hace el and entre esto
    TP = np.sum(np.logical_and(pred_labels == 255, true_labels == 255))

    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

    FP = np.sum(np.logical_and(pred_labels == 255, true_labels == 0))

    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 255))

    TT = 2 * TP

    TPR = np.round(TP*100/(TP+FN),2)
    FPR = np.round(FP*100/(FP+TN),2)
    F1 = np.round(TT*100/(TT+FP+FN),2)
    print(f"tpr: {TPR}, fpr: {FPR}, f1: {F1}")


if __name__=="__main__":
    # example = [14, 31, 4, 16]
    # adaptive = True
    # # for i in range(50):
    # for i in example:
    #     path = f'images/ISIC_00{num+i}.jpg'
    #     result = segmentation_img_gmm(path, a=adaptive, prints=True)
    #     seg_real = cv2.imread(f"images/ISIC_00{num+i}_segmentation.png",0)
    #     img_real = cv2.imread(path)
    #     cv2.destroyAllWindows()
    #     cv2.imshow(f"IMG original {i}", img_real)
    #     cv2.imshow("resultado", result)
    #     cv2.imshow("segmentacion real", seg_real)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     if adaptive:
    #         cv2.imwrite(f"results/IMG{num+i}GMMSA.jpg", result)
    #     else:
    #         cv2.imwrite(f"results/IMG{num+i}GMMS.jpg", result)
    # # Para correr imagenes del profe
    num = 24531
    for i in range(5):
        path = f'profe/ISIC_00{num+i}.jpg'
        segmentation_path = f"profe/ISIC_00{num+i}_segmentation.png"
        img_real = cv2.imread(path)
        real_segmentation = cv2.imread(segmentation_path, 0)
        result = segmentation_img_gmm(path, a = True, prints= True)
        print_results(real_segmentation, result)
        cv2.destroyAllWindows()
        cv2.imshow("imagen original", img_real)
        cv2.imshow("resultado", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
