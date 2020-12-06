import cv2
import numpy as np
from matplotlib import pyplot as plt
from adaptative import delete_min_areas
num = 24306


def printimg(text, img):
    cv2.imshow(text, img)
    
def kmeans(img, kind, canal = ""):
    img = cv2.medianBlur(img, 17)
    if kind == "RBG":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif kind == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif kind == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if canal != "":
        H, S, V = cv2.split(img)
        if canal == 1:
            printimg(f"canal {canal}", H)
            Z = H.flatten()
        elif canal == 2:
            printimg(f"canal {canal}", S)
            Z = S.flatten()
        elif canal == 3:
            printimg(f"canal {canal}", V)
            Z = V.flatten()
        Z = np.float32(Z)
    else:
        Z = img.flatten()
        Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


    # print(f"type_ret: {type(ret)}, type label: {type(label)} shape: {label.shape}, type center: {type(center)} shape: {center.shape} value: {center}")
    if canal != "":
        label = np.uint8(label*255)
        res2 = label.reshape((img.shape[0:2]))
    else:
        center = np.uint8(center)
        res = center[label.flatten()]
        res = res.reshape((img.shape))
        if kind == "RGB":
            res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        if kind == "LAB":
            res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        if kind == "HSV":
            res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        _, res2 = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    delete_min_areas(res2)
    auxiliar = 255-res2
    delete_min_areas(auxiliar)
    binary = 255-auxiliar
    count_1 = len(binary[binary == 255])
    count_0 = len(binary[binary == 0])
    if count_1 > count_0:
        aux = binary.copy()
        binary[aux == 255] = 0
        binary[aux == 0] = 255
    cv2.imshow('binary', binary)
    return binary

def make_results(kind, canal, view, write):
    for i in range(0, 50):
        img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
        segmentation = cv2.imread(f'images/ISIC_00{num+i}_segmentation.png')
        aux = np.hstack([img, segmentation])
        printimg("original", aux)
        result = kmeans(img, kind, canal)
        if view == True:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if write == True:
            cv2.imwrite(f"results/IMG{num+i}kmeans{kind}{canal}.jpg", result)
            
if __name__ == "__main__":
    # Para correr todas las pruebas
    """     kinds = ["RGB", "LAB", "HSV"] # HSV o RBG o LAB
        canal = "" # Del 1 al 3, vacio es ""
        aux = []
        for kind in kinds:
            make_results(kind, "", False, True)
            for i in range(1,4):
                make_results(kind, i , False, True) """
    # Para correr solo los mejores
    # (Ignorar las ventanas de cv2, terminara eventualmente aprox 30 seg)
    bests = ["RBG1", "HSV2", "LAB2"]
    for best in bests:
        kind, canal = best[0:3], int(best[3])
        make_results(best, canal, False, True)