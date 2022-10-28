import cv2
import numpy as np
 
img = cv2.imread('brokenEgg.jpeg')
img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
rows, cols = img.shape[:2]
 
# 원근 변환 전 후 4개 좌표
pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
pts2 = np.float32([[100, 50], [10, rows-50], [cols-100, 50], [cols-10, rows-50]])
 
# 변환 전 좌표를 원본 이미지에 표시
cv2.circle(img, (0, 0), 10, (255, 0, 0), -1)
cv2.circle(img, (0, rows), 10, (0, 255, 0), -1)
cv2.circle(img, (cols, 0), 10, (0, 0, 255), -1)
cv2.circle(img, (cols, rows), 10, (0, 255, 255), -1)
 
# 원근 변환 행렬 계산
mtrx = cv2.getPerspectiveTransform(pts1, pts2)
# 원근 변환 적용
output = cv2.warpPerspective(img, mtrx, (cols, rows))
 
result = np.hstack((img, output))
cv2.imshow('result', result)
cv2.imwrite('result.png', result)
cv2.waitKey()
