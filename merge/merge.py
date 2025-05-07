import cv2
import numpy as np

main_2F = cv2.imread("main_map_2F.png")
main_3F = cv2.imread("main_map_3F.png")

overay_2F = cv2.imread("overay_map_2F.png")
overay_3F = cv2.imread("overay_map_3F.png")

ratio = main_2F.shape[1] / main_2F.shape[0]
new_h = 2119 / ratio

main_2F = cv2.resize(main_2F,(main_3F.shape[1],int(new_h)))

merge = np.vstack((main_3F,main_2F))
cv2.imwrite("main_view.png",merge)
merge = cv2.resize(merge,(0,0),fx=0.5,fy=0.5)

ratio = overay_2F.shape[1] / overay_2F.shape[0]
new_h = 2119 / ratio

overay_2F = cv2.resize(overay_2F,(overay_3F.shape[1],int(new_h)))
merge2 = np.vstack((overay_3F,overay_2F))
cv2.imwrite("overay_view.png",merge2)
merge2 = cv2.resize(merge2,(0,0),fx=0.5,fy=0.5)

cv2.imshow("merge",merge)
cv2.imshow("merge2",merge2)
cv2.waitKey(0)