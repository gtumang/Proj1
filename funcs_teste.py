import cv2
import Funcoes as func
import sys
import numpy as np

def get_red_mask(img_hsv):
  __red_lower = np.array([0, 100, 50], dtype=np.uint8)
  __red_upper = np.array([10, 255, 255], dtype=np.uint8)
  __red_mask1 = cv2.inRange(img_hsv, __red_lower, __red_upper)
  __red_lower = np.array([170, 100, 50], dtype=np.uint8)
  __red_upper = np.array([180, 255, 255], dtype=np.uint8)
  __red_mask2 = cv2.inRange(img_hsv, __red_lower, __red_upper)
  red_mask = __red_mask1+__red_mask2
  return red_mask

def checa_cor(img,mask):
  img_filt = cv2.bitwise_and(img, img, mask=mask)
  if img_filt.sum()==0:
    return False
  return True