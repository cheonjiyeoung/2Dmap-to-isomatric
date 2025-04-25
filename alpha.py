import cv2
import numpy as np

iso_img = cv2.imread('isometric_map.png', cv2.IMREAD_UNCHANGED)

# 알파채널 추가
if iso_img.shape[2] == 3:
    b, g, r = cv2.split(iso_img)
    alpha = np.full_like(b, 190)
else:
    b, g, r, alpha = cv2.split(iso_img)
    alpha[:, :] = 190

# 누끼 따기
red_mask = (r == 0) & (g == 0) & (b == 0) # 빨간색 찾기
alpha[red_mask] = 0                         # 빨간색 투명처리
iso_img_rgba = cv2.merge((b, g, r, alpha))

cv2.imwrite('alpha.png', iso_img_rgba)
