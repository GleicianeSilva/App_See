import os
from glob import glob
import cv2
import pytesseract
from pytesseract import Output


def img_tratment(img):
    img_gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
    #cont = cv2.GaussianBlur(img_gray, (21, 21), 0)
    cont = cv2.medianBlur(img_gray, 5)
    cont = cv2.adaptiveThreshold(cont,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,3)
    #print(pytesseract.image_to_string(cont))
    return cont

def img_contours(img):
    edged = cv2.Canny(img, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    #cv2.imshow('img', img)
    print(img.shape)
    return img

def recognizer_coin(img):
    # If you don't have tesseract executable in your PATH, include the following:
    # pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>' like r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

    text = pytesseract.image_to_string(img, config='--psm 11')
    #print(text)
    return (f"Detected coin value is:{text}")
def get_imgs(path):
    return glob(os.path.join(os.getcwd(), path))
