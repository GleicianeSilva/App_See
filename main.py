import functions as f
import cv2
import functions_2 as f2

"""
Para funcionar o Pytesseract siga o que é dito nesse link
https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i
"""


g = f.get_imgs('data/0.01/0.1_0.png')

for img_file in g:

    img = f.img_tratment(img_file)
    contourn = f.img_contours(img)
    coin_number = f.recognizer_coin(contourn)
    convulcao = f2.quadradinho(contourn)
    cv2.imshow(coin_number, contourn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


