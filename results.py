import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
bad_imgs = [16, 17, 23, 25, 31,32, 39]

def calculate_restults(tipo):
    ## código sacado de : https://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/
    num = 24306
    # ESTO ES PARA TENER UAN IDEA DE COMO CLASIFICAN LAS IMAGENES "FACILES"=GOOD, "DIFICILES"=BAD Y EN TOTAL
    GOOD_T, GOOD_F = 0, 0
    BAD_T, BAD_F = 0, 0
    TOTAL_T, TOTAL_F = 0, 0
    total_tpr = []
    total_fpr = []
    bad_tpr = []
    bad_fpr = []
    for i in range(50):
        ## leer imagenes
        real = cv2.imread(f'images/ISIC_00{num+i}_segmentation.png', 0)
        obtenido = cv2.imread(f'results/IMG{num+i}{tipo}.jpg', 0)
        d=obtenido.flatten()
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
        if i in bad_imgs:
            BAD_T += TPR
            BAD_F += FPR
        else:
            GOOD_T += TPR
            GOOD_F += FPR

        TOTAL_T += TPR
        TOTAL_F += FPR
        total_tpr.append(TPR)
        total_fpr.append(FPR)
    
    total_tpr = np.array(total_tpr)
    total_fpr = np.array(total_fpr)
    worst_tpr = np.argsort(total_tpr)[:8] ## el - es para que sean los menores (los n menores tpr)
    worst_fpr = np.argsort(-total_fpr)[:8] ## sin el - es para que sean los mayores (los n mayores fpr)
    TOTAL_T_RES = np.round(TOTAL_T/50, 3)
    TOTAL_F_RES = np.round(TOTAL_F/50, 3)
    GOOD_T_RES = np.round(GOOD_T/(50-len(bad_imgs)), 3)
    GOOD_F_RES = np.round(GOOD_F/(50-len(bad_imgs)), 3)
    BAD_T_RES = np.round(BAD_T/len(bad_imgs), 3)
    BAD_F_RES = np.round(BAD_F/len(bad_imgs), 3)
    print(f"-----------------RESULTADOS {tipo}-------------")
    print("-------------- IMAGENES FÁCILES -----------------")
    print("TASA TPR: ", GOOD_T_RES, "TASA FPR: ", GOOD_F_RES)
    print("-------------- IMAGENES DIFICILES -----------------")
    print("TASA TPR: ", BAD_T_RES, "TASA FPR: ", BAD_F_RES)
    print("-------------------- TOTAL -----------------------")
    print("TASA TPR: ", TOTAL_T_RES, "TASA FPR: ", TOTAL_F_RES)
    return TOTAL_T_RES, TOTAL_F_RES, BAD_T_RES, BAD_F_RES, GOOD_T_RES, GOOD_F_RES, worst_tpr, worst_fpr


def resultados_csv(tipos, nombreoutput):
    data = []
    for tipo in tipos:
        tpr, fpr, tpr_bad, fpr_bad, tpr_good, fpr_good, worst_tpr, worst_fpr = calculate_restults(tipo)
        data.append([tipo, tpr, fpr, tpr_bad, fpr_bad, tpr_good, fpr_good, worst_tpr, worst_fpr])
    df = pd.DataFrame(data, columns=["Tipo", "TPR_FULL", "FPR_FULL", "TPR_BAD", "FPR_BAD", "TPR_GOOD", "FPR_GOOD", "8 peores tpr", "8 peores fpr"])
    df.to_csv("resultados.csv", sep=",", header=True, index=False)
    
if __name__ == "__main__":
    tipos = ["A", "O", "EQ", "EQO", "OS", "H", "W", "kmeansRBG1", "kmeansHSV2", "kmeansLAB2", "GMMSA", "GMMS"]
    for tipo in tipos:
        resultados_csv(tipo, "resultados.csv")
    # Para todos correr todos los kmeans.
    """ base = "kmeans"
        kinds = ["RGB", "HSV", "LAB"]
        aux = []
        for kind in kinds:
            aux.append(f"{base}{kind}")
            for i in range(1,4):
                aux.append(f"{base}{kind}{i}")
        resultados_csv(aux, "resultados_all_kmeans.csv") """
