import cv2
import Funcoes as func
import sys


path = sys.path[0]+'\\Imagens_DESENVOLVIMENTO\\'

img = cv2.imread(path+"_boa_02.png", cv2.IMREAD_COLOR).astype("uint8")
vermelho = func.acha_vermelhos(img)

print(func.analisa_vermelho(vermelho))
cv2.imshow("Image", vermelho)
cv2.waitKey(0)
