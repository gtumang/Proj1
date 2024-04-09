import cv2
import Funcoes as func
import sys
import numpy as np


path = sys.path[0]+'\\Imagens_DESENVOLVIMENTO\\'

img = cv2.imread(path+"quebrada_01.png", cv2.IMREAD_COLOR).astype("uint8")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


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

contours_1, _ = cv2.findContours(
    img_red_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours_1, -1, (255, 0, 0), 2)

print("Contornos vermelho: ", len(contours_1))
for contour in contours_1:
    print(cv2.contourArea(contour))
print('\n')

img_gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.equalizeHist(img_gray_raw)

ret1, img_bin = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)

sect = img_bin-img_red_bin

kernel = np.ones((5, 5), dtype=np.uint8)
sect_erode = cv2.erode(sect, kernel, iterations=2)
sect_out = cv2.dilate(sect_erode, kernel, iterations=2)

contours_2, _ = cv2.findContours(
    sect_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours_2, -1, (0, 255, 0), 2)

for contour in contours_2:
    print(cv2.contourArea(contour))

params = cv2.SimpleBlobDetector_Params()
# Set blob color (0=black, 255=white)
params.filterByColor = True
params.blobColor = 255
# Filter by Area
params.filterByArea = True
params.minArea = 27000
params.maxArea = 28100
# Filter by Circularity
params.filterByCircularity = False
# params.minCircularity = 0.45
# params.maxCircularity = 1.3
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.01
params.maxConvexity = 1
# Filter by Inertia
params.filterByInertia = False
# params.minInertiaRatio = 0.01
# params.maxInertiaRatio = 1
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

KP = detector.detect(sect_out)
print("Nro de blobs: ", len(KP))
sect_wkp = cv2.drawKeypoints(sect_out, KP, np.array([]), (255, 0, 0),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print('\n')

if len(contours_1) <= 14 and len(KP) == 1:
    print("Pilula boa")
else:
    print("Pilula inadequada")

cv2.imshow("Original", img)
# cv2.imshow("Grayscale", img_gray)
# cv2.imshow("Bin", img_bin)
cv2.imshow("Parte vermelha", img_red)
cv2.imshow("Parte vermelha bin", img_red_bin)
# cv2.imshow("Parte preta bin", sect)
# cv2.imshow("Parte preta", sect_out)
# cv2.imshow("Parte preta 2", sect_wkp)
cv2.waitKey(0)
