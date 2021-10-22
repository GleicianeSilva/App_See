import pre_processamento as f
import cv2
import random

"""
Para funcionar o Pytesseract siga o que é dito nesse link
https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i
"""

g = f.get_imgs('datasets/*jpg')

to_size = 320
scaled = []
scaled_labels = []


# Extraindo as moedas das imagens originais usada na definição extraido_moeda

for img_file in random.sample(g, 1):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)


  
    img = f.preprocessamento(img, to_size)
    if img is not None and len(img):
        scaled.append(img)
        scaled_labels.append(img_file.split('_')[0])   
        
    cv2.imshow('Moeda Brasileira', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()