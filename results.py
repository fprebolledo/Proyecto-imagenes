import cv2
import numpy as np
from matplotlib import pyplot as plt

def calculate_restults(tipo):
    ## código sacado de : https://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/
    num = 24306
    TOTAL_T, TOTAL_F = 0,0
    for i in range(50):
        ## leer imagenes
        real = cv2.imread(f'images/ISIC_00{num+i}_segmentation.png', 0)
        obtenido = cv2.imread(f'results/IMG{num+i}{tipo}.jpg', 0)
        
        # no entiendo porque los bordes cuando se guarda la imagen quedan con 1 o 254 y hay que hacer esto uwu
        _, obtenido = cv2.threshold(obtenido,127,255,cv2.THRESH_BINARY)
        true_labels, pred_labels = real.flatten(), obtenido.flatten()

        ## todas estas funciones devuelven un array de trues y falses si el valor coincide y luego se hace el and entre esto
        # como el and devuelve un 1 si ambos son iguales y un 0 si no, la suma nos da el total.
        
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = np.sum(np.logical_and(pred_labels == 255, true_labels == 255))
        
        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
        
        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(pred_labels == 255, true_labels == 0))
        
        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 255))
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        TOTAL_T += TPR
        TOTAL_F += FPR

    print("TASA TPR: ", np.round(TOTAL_T/50,3),"TASA FPR: ", round(TOTAL_F/50, 3))

if __name__ == "__main__":
    calculate_restults("A")