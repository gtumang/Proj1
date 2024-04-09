import cv2
import Funcoes as func
import sys
import numpy as np
from funcs_teste import *


path = sys.path[0]+'\\Imagens_DESENVOLVIMENTO\\'

img_original = cv2.imread(path+"quebrada_01.png",
                          cv2.IMREAD_COLOR).astype("uint8")

img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

img = img_original.copy()

# -------------------------------------Binarizando imagem inteira-------------------------------------------
img_gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.equalizeHist(img_gray_raw)

ret1, img_bin = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)




# -----------------------------------Achando parte vermelha com números destacados-----------------------------------
red_lower = np.array([0, 100, 50], dtype=np.uint8)
red_upper = np.array([10, 255, 255], dtype=np.uint8)
red_mask1 = cv2.inRange(img_hsv, red_lower, red_upper)
red_lower = np.array([170, 100, 50], dtype=np.uint8)
red_upper = np.array([180, 255, 255], dtype=np.uint8)
red_mask2 = cv2.inRange(img_hsv, red_lower, red_upper)
red_mask = red_mask1+red_mask2

img_red = cv2.bitwise_and(img, img, mask=red_mask)
img_red_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
ret2, img_red_bin = cv2.threshold(img_red_gray, 1, 255, cv2.THRESH_BINARY)

red_part_mask = np.zeros((img.shape[1], img.shape[0]), np.uint8)

# -------------------------------Achando contorno externo da parte vermelha-------------------------------------------
contours_1, _ = cv2.findContours(
    img_red_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours_1, -1, (255, 0, 0), 2)

# -----------------------------------------Checa se a cor está certa------------------------------------------------------
# if len(contours_1) == 0:
#     print("Cor errada")
#     sys.exit()

# ---------------------------------Acha parte vermelha sem destacar os números--------------------------------------------
cv2.drawContours(red_part_mask, contours_1, -1, (255, 255, 255), -1)
red_part = cv2.bitwise_and(img_original, img_original, mask=red_part_mask)

red_part_gray = cv2.cvtColor(red_part, cv2.COLOR_BGR2GRAY)

ret3, red_part_bin = cv2.threshold(red_part_gray, 180, 255, cv2.THRESH_BINARY)

# ----------------------------------Checa se parte vermelha está quebrada-------------------------------------------------
# if red_part_bin.sum() != 0:
#     print("Pilula quebrada")
#     sys.exit()

x_red, y_red, w_red, h_red = cv2.boundingRect(contours_1[len(contours_1)-1])
cv2.putText(img, "w: "+str(w_red), (x_red, y_red - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
cv2.putText(img, "h: "+str(h_red), (x_red, y_red - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
cv2.rectangle(img, (x_red, y_red),
              (x_red + w_red, y_red + h_red), (36, 255, 12), 1)

# ----------------------------------------Tratando seção preta------------------------------------------------
sect = img_bin-img_red_bin

kernel = np.ones((5, 5), dtype=np.uint8)
sect_erode = cv2.erode(sect, kernel, iterations=2)
sect_out = cv2.dilate(sect_erode, kernel, iterations=2)

contours_2_raw, _ = cv2.findContours(
    sect_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours_2 = sorted(contours_2_raw, key=lambda x: cv2.contourArea(x))
cv2.drawContours(img, contours_2, len(contours_2)-1, (0, 255, 0), 2)

sect_preta_orig_mask = np.zeros((img.shape[1], img.shape[0]), np.uint8)
cv2.drawContours(sect_preta_orig_mask, contours_2,
                 len(contours_2)-1, (255, 255, 255), -1)

sect_preta_orig = cv2.bitwise_and(
    img_original, img_original, mask=sect_preta_orig_mask)
ret4, sect_preta_orig_bin = cv2.threshold(
    sect_preta_orig, 160, 255, cv2.THRESH_BINARY)

print(cv2.contourArea(contours_2[len(contours_2)-1]))

# if sect_preta_orig_bin.sum() != 0 or cv2.contourArea(contours_2[len(contours_2)-1]) < 27000 or cv2.contourArea(contours_2[len(contours_2)-1]) > 29000:
#     print("Pilula quebrada")
#     sys.exit()

# ---------------------------------------Achando altura e largura da seção preta----------------------------------

x_sect, y_sect, w_sect, h_sect = cv2.boundingRect(
    contours_2[len(contours_2)-1])
cv2.putText(img, "w: "+str(w_sect), (x_sect, y_sect - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
cv2.putText(img, "h: "+str(h_sect), (x_sect, y_sect - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
cv2.rectangle(img, (x_sect, y_sect), (x_sect + w_sect,
              y_sect + h_sect), (36, 255, 12), 1)

h_pill = max([h_red, h_sect])
w_pill = w_red+w_sect

x_sect = x_sect+w_sect/2
y_sect = y_sect+y_sect/2

x_red = x_red+w_red/2
y_red = y_red+y_red/2

center = (np.uint8((x_red+x_sect)/2), np.uint8((y_red+y_sect)/2))
print(center)
cv2.circle(img, center, 6, (200, 0, 0), -1)

print("h1: ", h_sect)
print("h2: ", h_red)
print("w: ", w_red+w_sect)

# cv2.imshow("Original", img)
# cv2.imshow("Grayscale", img_gray)
# cv2.imshow("Bin", img_bin)
cv2.imshow("Parte vermelha", img_red)
# cv2.imshow("Parte vermelha 2", red_part)
# cv2.imshow("Parte vermelha grayscale 2", red_part_gray)
# cv2.imshow("Parte vermelha bin", img_red_bin)
# cv2.imshow("Parte vermelha bin 2", red_part_bin)
cv2.imshow("Parte preta bin", sect)
cv2.imshow("Parte preta", sect_out)
cv2.imshow("Parte preta original", sect_preta_orig)
cv2.imshow("Parte preta original bin", sect_preta_orig_bin)
cv2.waitKey(0)
