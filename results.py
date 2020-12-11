import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
bad_imgs = [16, 17, 23, 25, 31,32, 39]

def calculate_results(tipo, cantidad, inicio):
    ## código sacado de : https://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/
    num = inicio
    # ESTO ES PARA TENER UAN IDEA DE COMO CLASIFICAN LAS IMAGENES "FACILES"=GOOD, "DIFICILES"=BAD Y EN TOTAL
    GOOD_T, GOOD_F, GOOD_F1 = 0, 0, 0
    BAD_T, BAD_F, BAD_F1 = 0, 0, 0
    
    TOTAL_T, TOTAL_F, TOTAL_F1 = 0, 0, 0
    total_tpr = []
    total_f1 = []
    total_fpr = []
    bad_tpr = []
    bad_fpr = []
    for i in range(cantidad):
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
        
        TT = 2 * TP
        
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        F1 = TT/(TT+FP+FN)
        
        if i in bad_imgs:
            BAD_T += TPR
            BAD_F += FPR
            BAD_F1 += F1
        else:
            GOOD_T += TPR
            GOOD_F += FPR
            GOOD_F1 += F1
        TOTAL_T += TPR
        TOTAL_F += FPR
        TOTAL_F1 += F1
        total_tpr.append(TPR)
        total_fpr.append(FPR)
        total_f1.append(F1)
    
    total_tpr = np.array(total_tpr)
    total_fpr = np.array(total_fpr)
    # total_f1 = np.array(total_f1)
    # best_f1 = np.argsort(-total_f1)[0]
    # worst_f1 = np.argsort(total_f1)[0]
    worst_tpr = np.argsort(total_tpr)[:8] ## el - es para que sean los menores (los n menores tpr)
    worst_fpr = np.argsort(-total_fpr)[:8] ## sin el - es para que sean los mayores (los n mayores fpr)
    TOTAL_T_RES = np.round((TOTAL_T/50) *100, 3)
    TOTAL_F_RES = np.round((TOTAL_F/50) *100, 3)
    TOTAL_F1_RES = np.round((TOTAL_F1/50)*100, 3)
    GOOD_T_RES = np.round(GOOD_T*100/(50-len(bad_imgs)), 3)
    GOOD_F_RES = np.round(GOOD_F*100/(50-len(bad_imgs)), 3)
    GOOD_F1_RES = np.round(GOOD_F1*100/(50-len(bad_imgs)), 3)
    BAD_T_RES = np.round(BAD_T*100/len(bad_imgs), 3)
    BAD_F_RES = np.round(BAD_F*100/len(bad_imgs), 3)
    BAD_F1_RES = np.round(BAD_F1*100/len(bad_imgs), 3)
    # print(f"-----------------RESULTADOS {tipo}-------------")
    # print("-------------- IMAGENES FÁCILES -----------------")
    # print("TASA TPR: ", GOOD_T_RES, "TASA FPR: ", GOOD_F_RES)
    # print("-------------- IMAGENES DIFICILES -----------------")
    # print("TASA TPR: ", BAD_T_RES, "TASA FPR: ", BAD_F_RES)
    # print("-------------------- TOTAL -----------------------")
    # print("TASA TPR: ", TOTAL_T_RES, "TASA FPR: ", TOTAL_F_RES)
    # print(best_f1)
    # print(f"{total_f1[best_f1]}")
    # print(worst_f1)
    # print(f"{total_f1[worst_f1]}")
    return TOTAL_T_RES, TOTAL_F_RES, TOTAL_F1_RES, BAD_T_RES, BAD_F_RES, BAD_F1_RES, GOOD_T_RES, GOOD_F_RES, GOOD_F1_RES, worst_tpr, worst_fpr


def resultados_csv(tipos, output_name):

    columnas = [
        "Tipo",
        "TPR_FULL",
        "FPR_FULL",
        "F1_FULL",
        "TPR_BAD",
        "FPR_BAD",
        "F1_BAD",
        "TPR_GOOD",
        "FPR_GOOD",
        "F1_GOOD",
        "8 peores tpr",
        "8 peores fpr"
    ]

    data = []
    for tipo in tipos:
        seg_results = calculate_results(tipo)
        data.append([tipo, *seg_results])
    
    df = pd.DataFrame(data, columns=columnas)
    df.to_csv(output_name, sep=",", header=True, index=False)
    
if __name__ == "__main__":
    # Tienen que estar todos los tipos, si no, tira error al no encontrar imagen.
    tipos = ["A", "O", "EQ", "EQO", "GMMS", "GMMSA","OS", "H", "W", "RW", "kmeansRGB1", "kmeansHSV2", "kmeansLAB2"]
    # tipos = ["kmeansRGB1", "kmeansHSV2", "kmeansLAB2"]
    resultados_csv(tipos, "resultados.csv")
    # Para todos correr todos los kmeans.

