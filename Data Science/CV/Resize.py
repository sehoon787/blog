import cv2

src = cv2.imread('brokenEgg.jpeg')
y, x, c = src.shape
print("x: " + str(x) + ", y: " + str(y) + ", channel: " + str(c))


def upsampling(src):
    INTER_NEAREST = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_NEAREST)
    INTER_LINEAR = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_LINEAR)
    INTER_AREA = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_AREA)
    INTER_CUBIC = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_CUBIC)
    INTER_LANCZOS4 = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow('src', src[int(500/(1920/y)):int(500/(1920/y))+350, int(400/(1280/x)):int(400/(1280/x))+350])
    cv2.imshow('INTER_NEAREST', INTER_NEAREST[500:900, 400:800])
    cv2.imshow('INTER_LINEAR', INTER_LINEAR[500:900, 400:800])
    cv2.imshow('INTER_AREA', INTER_AREA[500:900, 400:800])
    cv2.imshow('INTER_CUBIC', INTER_CUBIC[500:900, 400:800])
    cv2.imshow('INTER_LANCZOS4', INTER_LANCZOS4[500:900, 400:800])
    cv2.waitKey()
    cv2.destroyAllWindows()


def downsampling(src):
    INTER_NEAREST = cv2.resize(src, (380, 480), interpolation=cv2.INTER_NEAREST)
    INTER_LINEAR = cv2.resize(src, (380, 480), interpolation=cv2.INTER_LINEAR)
    INTER_AREA = cv2.resize(src, (380, 480), interpolation=cv2.INTER_AREA)
    INTER_CUBIC = cv2.resize(src, (380, 480), interpolation=cv2.INTER_CUBIC)
    INTER_LANCZOS4 = cv2.resize(src, (380, 480), interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow('INTER_NEAREST', INTER_NEAREST)
    cv2.imshow('INTER_LINEAR', INTER_LINEAR)
    cv2.imshow('INTER_AREA', INTER_AREA)
    cv2.imshow('INTER_CUBIC', INTER_CUBIC)
    cv2.imshow('INTER_LANCZOS4', INTER_LANCZOS4)
    cv2.waitKey()
    cv2.destroyAllWindows()


upsampling(src)
downsampling(src)
