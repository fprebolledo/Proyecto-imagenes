import cv2
import numpy as np
from matplotlib import pyplot as plt
from adaptative import delete_min_areas
num = 24306


def printear_img_plt(img, text, fin=False):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    fig.suptitle(text, fontsize=16)
    if fin == True:
        plt.show()
        
def tophat(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((30, 30), np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    to_rbg = cv2.cvtColor(blackhat, cv2.COLOR_HSV2RGB)
    to_gray = cv2.cvtColor(to_rbg, cv2.COLOR_RGB2GRAY)
    h ,s ,v = cv2.split(blackhat)
    printear_img_plt(h, "h", False)
    # printear_img_plt(s, "s", False)
    # printear_img_plt(v, "v", False)

    threshold, binary = cv2.threshold(h, 150, 255, cv2.THRESH_BINARY)
    delete_min_areas(binary)
    # img_new[binary == (0,0,0)] = 255
    # printear_img_plt(img, "IMG", False)
    printear_img_plt(blackhat, "BLACKHAT", False)
    # printear_img_plt(tophat, "TOPHAT", False)
    # printear_img_plt(img_new, "XD", False)
    printear_img_plt(binary, "thresh", True)

def contrast_img(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # cv2.imshow("lab", lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imshow('final', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final

def xd(img):
    imgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8, 8))
    imgray_ad = clahe.apply(imgr)  # adaptive
    imgray = cv2.equalizeHist(imgr)  # global
    res = np.hstack((imgray, imgray_ad))  # so we can plot together

    plt.imshow(res, cmap='gray')
    plt.show()

    ret, thresh = cv2.threshold(imgray_ad, 20, 255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    plt.imshow(thresh, cmap='gray')
    plt.show()
    
if __name__ == "__main__":
    i = 3
    img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
    # xd(img)
    # img = contrast_img(img)
    tophat(img)
    # conversion_contrast(img)
    
