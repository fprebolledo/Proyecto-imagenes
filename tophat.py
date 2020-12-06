from scipy.optimize import minimize
from skimage.transform import resize
import cv2
import numpy as np
from matplotlib import pyplot as plt
from adaptative import delete_min_areas

num = 24306


_k = np.ones(3)


def rgb2hcm(image):
    '''\
    rgb2hcm(image)

    Segmentation of an object with homogeneous background.

    Parameters
    ----------
    image: 3 dimensional ndarray
        The RGB input image.

    Returns
    -------
    hcm: 2 dimensional ndarray
        high contrast grayscale representation of input image


    Examples
    --------
    (TODO)
    '''
    if image.ndim < 3:
        I = image
    else:
        img_resize = resize(image, (64, 64), order=3,
                            mode='reflect', anti_aliasing=False)
        k = minimize(monochrome_std, [1., 1.], args=(img_resize,))['x']
        _k[:2] = k
        I = image @ _k
    J = I - I.min()
    J = J / J.max()
    n = J.shape[0] // 4
    m = J.shape[1] // 4

    if (J[:n, :m].mean() > .4):
        J = 1 - J
    return J


def monochrome_std(k, image):
    _k[:2] = k
    I = image @ _k
    return - I.std(ddof=1) / (I.max() - I.min())

def printear_img_plt(img, text, fin=False):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    fig.suptitle(text, fontsize=16)
    if fin == True:
        plt.show()
def printimg(text, img):
    cv2.imshow(text, img)
    
def tophat(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((30, 30), np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    to_rbg = cv2.cvtColor(blackhat, cv2.COLOR_HSV2RGB)
    to_gray = cv2.cvtColor(to_rbg, cv2.COLOR_RGB2GRAY)
    h ,s ,v = cv2.split(blackhat)
    printimg('tophat', h)

    # printear_img_plt(s, "s", False)
    # printear_img_plt(v, "v", False)

    threshold, binary = cv2.threshold(h, 150, 255, cv2.THRESH_BINARY)
    delete_min_areas(binary)
    printimg('tophat', binary)

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



if __name__ == "__main__":
    for i in range(0, 50):
        img = cv2.imread(f'images/ISIC_00{num+i}.jpg')
        imggray = rgb2hcm(img)
        cv2.imshow("imggray", imggray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
