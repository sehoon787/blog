import cv2
import numpy as np

def flip(img):
    rows, cols = img.shape[:2]

    # 뒤집기 변환 행렬
    mflip = np.float32([[-1, 0, cols - 1], [0, -1, rows - 1]])  # 변환 행렬 생성
    fliped1 = cv2.warpAffine(img, mflip, (cols, rows))  # 변환 적용

    # remap 함수로 뒤집기
    mapy, mapx = np.indices((rows, cols), dtype=np.float32)  # 매핑 배열 초기화 생성
    mapx = cols - mapx - 1  # x축 좌표 뒤집기 연산
    mapy = rows - mapy - 1  # y축 좌표 뒤집기 연산
    fliped2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)  # remap 적용

    return fliped1, fliped2

def waveDistortion(img, amp=20, waveFreq=32):
    h, w = img.shape[:2]  # 입력 영상의 높이와 넓이 정보 추출

    # np.indice는 행렬의 인덱스값 x좌표값 y좌표값을 따로따로 행렬의 형태로 변환해줌
    mapy, mapx = np.indices((h, w), dtype=np.float32)

    # borderMode는 근방의 색깔로 대칭되게 해서 채워줌, 기본값은 빈 공간을 검은색으로 표현
    # sin, cos 함수를 적용한 변형 매핑 연산
    sinx = mapx + amp * np.sin(mapy / waveFreq)
    cosy = mapy + amp * np.cos(mapx / waveFreq)

    img_x = cv2.remap(img, sinx, mapy, cv2.INTER_LINEAR)  # x축만 sin 곡선 적용
    img_y = cv2.remap(img, mapx, cosy, cv2.INTER_LINEAR)  # y축만 cos 곡선 적용
    img_both = cv2.remap(img, sinx, cosy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)

    return img_x, img_y, img_both

def LensDistortionImage(img, exp=2, scale=1):
    '''
    :param img: image
    :param exp: 오목, 볼록 지수 (오목 : 0.1 ~ 1, 볼록 : 1.1~) => 1보다 작으면 오목 렌즈 효과를 내고, 1보다 크면 볼록 렌즈 효과
    :param scale: 변환 영역 크기 (0 ~ 1)
    '''
    rows, cols = img.shape[:2]

    # 매핑 배열 생성
    mapy, mapx =np.indices((rows, cols), dtype=np.float32)

    # 좌상단 기준좌표에서 -1~1로 정규화된 중심점 기준 좌표로 변경
    mapx = 2 * mapx / (cols - 1) - 1
    mapy = 2 * mapy / (rows - 1) - 1

    # 직교좌표를 극 좌표로 변환 ---④
    r, theta = cv2.cartToPolar(mapx, mapy)

    # 왜곡 영역만 중심확대/축소 지수 적용
    r[r < scale] = r[r < scale] ** exp

    # 극 좌표를 직교좌표로 변환
    mapx, mapy = cv2.polarToCart(r, theta)

    # 중심점 기준에서 좌상단 기준으로 변경
    mapx = ((mapx + 1) * cols - 1) / 2
    mapy = ((mapy + 1) * rows - 1) / 2
    # 재매핑 변환
    result = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    return result


img = cv2.imread('brokenEgg.jpeg')

concave = LensDistortionImage(img, exp=0.5)
convex = LensDistortionImage(img, exp=2)
result = np.hstack((img, concave, convex))
cv2.imshow('result', result)
cv2.imwrite('result.png', result)
cv2.waitKey()
