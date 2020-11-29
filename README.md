# Proyecto: Segmentacion de Lunares

## Enunciado

El algoritmo disenado debe entregar una imagen binaria, y a la vez debe compararse con la segmentacion ideal entregando por imagen los valores:

* TP (true positives): número de pixeles pertenecientes al lunar correctamente segmentados
* TN (true negatives): número de pixeles pertenecientes al fondo (piel) correctamente segmentados
* FP (false positives): número de pixeles pertenecientes al fondo (piel) segmentados como lunar
* FN (false negatives): número de pixeles pertenecientes al lunar segmentados como fondo (piel)

Asimismo, se debe calcular el TPR (tasa de true positives) y el FPR (tasa de false positives) definidos respectivamente como TP/(TP+FN) y FP/(FP+TN), que idealmente deben ser 100% y 0%. 

Esta permitido usar librerias clasicas de procesamiento de imagenes, pero no de machine learning. Todo lo que se use deben saber explicarlo.

## Formato de imagenes de resultado:
### Terminaciones:
- O: threshold otsu
- EQ: ecualizacion (guassiano ?)
- EQO: ecualizacion 2 con otsu
- OS: Otsu + correcion de sombras con umbral
- A: adaptativa con media
- W: sobel + watershed
- H: histogram umbral