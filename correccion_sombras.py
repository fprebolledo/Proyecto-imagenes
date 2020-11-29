import cv2
import numpy as np
from matplotlib import pyplot as plt
from adaptative import delete_min_areas

num = 24306

for i in range(50):
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    median = cv2.medianBlur(img,21)
    blur = cv2.GaussianBlur(median,(7,7),0)
    
    grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)


    mean = np.mean(grey) * 1.2
    print(mean)
    
    rows, cols = img.shape[0], img.shape[1]
    new_gray = np.ones((rows,cols))
    for row in range(rows):
        for col in range(cols):
            if grey[row][col]<mean:
                grey[row][col] -= 12
            else:
                grey[row][col] += 12
           
    _, binary = cv2.threshold(grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    delete_min_areas(binary)
    aux = 255-binary
    delete_min_areas(aux)
    binary = 255-aux
    
    cv2.imwrite(f"results/IMG{num+i}OS.jpg",binary)