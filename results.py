letras = ["A", "B", "C", "M","ME", "G"]
import cv2
import numpy as np
from matplotlib import pyplot as plt

num = 24306

for i in range(50):
    real = cv2.imread(f'images/ISIC_00{num+0}_segmentation.png')
    obtenido = cv2.imread(f'results/IMG{num+i}ME.jpg')
    true_labels, pred_labels = real.flatten(), obtenido.flatten()

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 255, true_labels == 255))
    
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 255, true_labels == 0))
    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 255))
    print(TP, TN, FP, FN)
    break