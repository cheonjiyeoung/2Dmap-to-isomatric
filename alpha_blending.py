import cv2
import numpy as np

path1 = "keti_3f.png"
path2 = "alpha.png"

img1 = cv2.imread(path1)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2BGRA)
img2 = cv2.imread(path2,cv2.IMREAD_UNCHANGED)

alpha = 0.3
beta = 0.7

h,w,c = img1.shape

img2 = cv2.resize(img2,(w,h))

print(img1.shape)
print(img2.shape)

img3 = (img1 * alpha + img2 * beta).astype(np.uint8)

cv2.imshow("blended",img3)
cv2.waitKey(0)