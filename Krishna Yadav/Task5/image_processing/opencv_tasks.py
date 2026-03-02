import cv2
import numpy as np

# Load sample image (we will use one MNIST image)
image = cv2.imread("sample_digit.png", 0)

# Edge Detection (Canny)
edges = cv2.Canny(image,100,200)
cv2.imwrite("edges.png", edges)

# Thresholding
_, thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
cv2.imwrite("threshold.png", thresh)

# Augmentation - Flip
flip = cv2.flip(image,1)
cv2.imwrite("flip.png", flip)

# Augmentation - Rotate
(h,w) = image.shape
matrix = cv2.getRotationMatrix2D((w/2,h/2),45,1)
rotate = cv2.warpAffine(image,matrix,(w,h))
cv2.imwrite("rotate.png", rotate)

# Brightness Adjustment
bright = cv2.convertScaleAbs(image, alpha=1, beta=50)
cv2.imwrite("bright.png", bright)

print("OpenCV Tasks Completed!")