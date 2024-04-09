from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import cv2
import Funcoes as func
import sys

path = sys.path[0]+'\\Imagens_DESENVOLVIMENTO\\'

image_referencia = cv2.imread(
    cv2.samples.findFile(path+"_boa_01.png"), cv2.IMREAD_COLOR).astype("uint8")
image_colorida_v = cv2.imread(
    cv2.samples.findFile(path+"color1.png"), cv2.IMREAD_COLOR).astype("uint8")
image_colorida_a = cv2.imread(
    cv2.samples.findFile(path+"color3.png"), cv2.IMREAD_COLOR).astype("uint8")
image_amassada = cv2.imread(cv2.samples.findFile(
    path+"_boa_02.png"), cv2.IMREAD_COLOR).astype("uint8")

verde = func.acha_verdes(image_colorida_v)
azul = func.acha_azuis(image_colorida_a)
vermelho = func.acha_vermelhos(image_referencia)

valores_imagem_referencia = func.encontra_tamanhos(image_referencia)
orig = valores_imagem_referencia[0]
width_referencia = valores_imagem_referencia[1]
height_referencia = valores_imagem_referencia[2]

amassada = func.encontra_tamanhos(image_amassada)
width_amassada = amassada[1]
height_amassada = amassada[2]

print(height_amassada, height_referencia)
print(func.analisa_vermelho(vermelho))
print(func.analisa_amassada(width_referencia,
      height_referencia, width_amassada, height_amassada))

# show the output image
cv2.imshow("Image", image_referencia)
cv2.imshow("Image 2", image_amassada)
cv2.waitKey(0)
