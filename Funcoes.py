import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance as dist
from imutils import perspective, contours, grab_contours, is_cv2
import numpy as np


def nmr_de_blobs(thresh):

    img1_eros = thresh
    kernel = np.ones((5, 5), np.uint8)
    img1_eros = cv2.erode(img1_eros, kernel, iterations=5)
    img1_eros = cv2.dilate(img1_eros, kernel, iterations=4)

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Set blob color (0=black, 255=white)
    params.filterByColor = True
    params.blobColor = 0

    # Filter by Area
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 20000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.maxCircularity = 1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.maxConvexity = 1

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs

    KP = detector.detect(img1_eros)
    nmr = len(KP)

    return nmr


def acha_azuis(img):
    ret, mask = cv2.threshold(img[:, :, 0], 100, 255, cv2.THRESH_BINARY)

    mask3 = np.zeros_like(img)
    mask3[:, :, 0] = mask
    mask3[:, :, 1] = mask
    mask3[:, :, 2] = mask

    # extracting `orange` region using `bitewise_and`
    blue = cv2.bitwise_and(img, mask3)

    hsv = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h_mask = cv2.inRange(s, 0, 100)

    kernel = np.ones((3, 3), np.uint8)
    h_mask = cv2.dilate(h_mask, kernel, iterations=1)
    # h_mask = cv2.erode(h_mask, kernel, iterations = 3)

    return h_mask


def acha_verdes(img):
    ret, mask = cv2.threshold(img[:, :, 1], 100, 255, cv2.THRESH_BINARY)

    mask3 = np.zeros_like(img)
    mask3[:, :, 0] = mask
    mask3[:, :, 1] = mask
    mask3[:, :, 2] = mask

    # extracting `orange` region using `bitewise_and`
    green = cv2.bitwise_and(img, mask3)

    hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h_mask = cv2.inRange(s, 0, 100)

    kernel = np.ones((3, 3), np.uint8)
    h_mask = cv2.dilate(h_mask, kernel, iterations=1)
    # h_mask = cv2.erode(h_mask, kernel, iterations = 3)

    return h_mask


def acha_vermelhos(img):
    ret, mask = cv2.threshold(img[:, :, 2], 100, 255, cv2.THRESH_BINARY)

    mask3 = np.zeros_like(img)
    mask3[:, :, 0] = mask
    mask3[:, :, 1] = mask
    mask3[:, :, 2] = mask

    # extracting `orange` region using `bitewise_and`
    green = cv2.bitwise_and(img, mask3)

    hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h_mask = cv2.inRange(s, 0, 100)

    kernel = np.ones((3, 3), np.uint8)
    h_mask = cv2.dilate(h_mask, kernel, iterations=1)
    # h_mask = cv2.erode(h_mask, kernel, iterations = 3)

    return h_mask


def filtro_media(m, img):
    (h, w) = img.shape

    #### KERNEL ####

    # Kernel creation

    d = int((m-1)/2)
    kernel = np.ones((m, m), dtype="int16")/m**2
    # print("kernel_x: \n", kernel)

    # Init processed figure
    fig_out = np.zeros((h, w), dtype="uint8")

    # Image reading

    for i in range(d, h-d):
        for j in range(d, w-d):
            secao_img = img[i-d:i+d+1, j-d:j+d+1]

            prod_img_ker = kernel * secao_img
            somatorio = prod_img_ker.sum()

            fig_out[i, j] = somatorio

    return fig_out


def filtro_mediana(m, img):
    (h, w) = img.shape

    #### KERNEL ####

    # Kernel creation

    d = int((m-1)/2)
    kernel = np.ones((m, m), dtype="int16")/m**2
    # print("kernel_x: \n", kernel)

    # Init processed figure
    fig_out = np.zeros((h, w), dtype="uint8")

    # Image reading

    for i in range(d, h-d):
        for j in range(d, w-d):
            secao_img = img[i-d:i+d+1, j-d:j+d+1]

            prod_img_ker = kernel * secao_img
            mediana = np.median(prod_img_ker)

            fig_out[i, j] = mediana

    return fig_out


def filtro_prewitt_sobel_abs(m, img, kernel):
    (h, w) = img.shape

    #### KERNEL ####

    # Kernel creation

    d = int((m-1)/2)
    # print("kernel_x: \n", kernel)

    # Init processed figure
    fig_out = np.zeros((h, w), dtype="uint8")

    # Image reading

    for i in range(d, h-d):
        for j in range(d, w-d):
            secao_img = img[i-d:i+d+1, j-d:j+d+1]

            prod_img_ker = kernel * secao_img
            somatorio = prod_img_ker.sum()

            fig_out[i, j] = abs(somatorio)

    return fig_out


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def encontra_tamanhos(image):  # COLORIDA #

    # USAGE
    # python object_size.py --image images/example_01.png --width 0.955
    # python object_size.py --image images/example_02.png --width 0.955
    # python object_size.py --image images/example_03.png --width 3.5

    # import the necessary packages

    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    # 	help="path to the input image")
    # ap.add_argument("-w", "--width", type=float, required=True,
    # 	help="width of the left-most object in the image (in inches)")
    args = 1

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 170)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        (width, height) = box[1]
        box = cv2.cv.BoxPoints(box) if is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / args

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

    return orig, width, height


def analisa_vermelho(estampa1):
    (h, w) = estampa1.shape

    ret, thresh1 = cv2.threshold(estampa1, 90, 255, 0)

    kernel = np.ones((3, 3), np.uint8)
    img1_eros = cv2.dilate(thresh1, kernel, iterations=2)
    img1_eros = cv2.erode(thresh1, kernel, iterations=7)

    # Kernel creation

    kernel_mascara_laplace = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype="int16")

    fig_out = filtro_prewitt_sobel_abs(3, thresh1, kernel_mascara_laplace)

    img1_eros = fig_out

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Set blob color (0=black, 255=white)
    params.filterByColor = True
    params.blobColor = 0

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs

    KP = detector.detect(img1_eros)
    flag = False

    if len(KP) != 0:
        flag = True

    return flag


def analisa_amassada(width_ref, height_ref, width_am, height_am):
    if width_am <= width_ref+4 and height_am <= height_ref+4 and width_am > width_ref-4 and height_am > height_ref-4:
        return True
    else:
        return False
