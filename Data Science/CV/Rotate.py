import cv2
import math
import numpy as np

img = cv2.imread('brokenEgg.jpg')

rad = 20 * math.pi / 180    # 각도 설정

def normalRotate(img, angle):
    # np.array로 Affine 행렬 생성
    affne = np.array([[math.cos(angle), math.sin(angle), 0],
                    [-math.sin(angle), math.cos(angle), 0]], dtype=np.float32)

    result = cv2.warpAffine(img, affne, (0, 0))

    return result

def RotateImage(img, angle, scale=1):
    if img.ndim > 2:
        height, width, channel = img.shape
    else:
        height, width = img.shape

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    result = cv2.warpAffine(img, matrix, (width, height))

    return result

result = RotateImage(img, rad, scale=0.6)

cv2.imshow('original', img)
cv2.imshow('result', result)
cv2.waitKey(0)
