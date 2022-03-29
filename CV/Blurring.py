import cv2
import numpy as np

def BlurImage(img, option=0, n=3):
    '''
    :param img: original image
    :param option: 0: Convolution, 1: Averaging Blurring, 2: Gaussian Blurring, 3: Median Blurring, 4: Bilateral Filtering
    :param n: size
    '''
    if option == 0:
        # 컨볼루션 계산은 커널과 이미지 상에 대응되는 값끼리 곱한 후, 모두 더하여 구함
        # 이 결과값을 결과 영상의 현재 위치에 기록
        # default 3
        kernel = np.ones((n, n), np.float32) / n**2
        result = cv2.filter2D(img, -1, kernel)
    elif option == 1:
        # 이웃 픽셀의 평균을 결과 이미지의 픽셀값으로하는 평균 블러링
        # 에지 포함해서 전체적으로 블러링
        # default 15
        result = cv2.blur(img, (n, n))
    elif option == 2:
        # 모든 픽셀에 똑같은 가중치를 부여했던 평균 블러링과 달리 가우시안 블러링은 중심에 있는 픽셀에 높은 가중치를 부여
        # 캐니(Canny)로 에지를 검출하기전에 노이즈를 제거하기 위해 사용
        # default 15
        result = cv2.GaussianBlur(img, (n, n), 0)
    elif option == 3:
        # 관심화소 주변으로 지정한 커널 크기(5 x 5) 내의 픽셀을 크기순으로 정렬한 후 중간값을 뽑아서 픽셀값으로 사용
        # default 15
        result = cv2.medianBlur(img, n)
    elif option == 4:
        # 에지를 보존하면서 노이즈를 감소시킬수 있는 방법
        # default 15
        result = cv2.bilateralFilter(img, n, 75, 75)
    return result

img = cv2.imread('brokenEgg.jpg')
result = BlurImage(img, option=0, n=15)

merged = np.hstack((img, result))
cv2.imshow('original-result', merged)
cv2.waitKey(0)
