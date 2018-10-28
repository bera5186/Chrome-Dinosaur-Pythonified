import numpy as np
import cv2

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()
    h = rect.bottom()

    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68,2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords
def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
    return resized,ratio



