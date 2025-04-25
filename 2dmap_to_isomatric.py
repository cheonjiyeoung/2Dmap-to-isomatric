import cv2
import numpy as np

map_img = cv2.imread('map3.png')
h, w = map_img.shape[:2]

# 아이소메트릭 회전 + 스케일 행렬
# 평면 이미지 -> 아이소메트릭으로 변환시 y축 1/2, 45도 회전 적용됨
def get_isometric_matrix():
    angle = np.deg2rad(45)
    scale_y = 0.5
    M = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle) * scale_y, np.cos(angle) * scale_y, 0]
    ], dtype=np.float32)
    return M

M = get_isometric_matrix()

# 변환 후 이미지 크기 (여유 공간 확보)
output_w = int(w + h)
output_h = int((w + h) * 0.5)
output_size = (output_w, output_h)

# 중심 정렬을 위한 이동 보정 (출력 이미지 중앙으로 이동)
center = np.array([w / 2, h / 2])                                       # 원본 이미지 중심 좌표
center_transformed = M[:, :2] @ center.reshape(2, 1)                    # 중심을 변환 행렬에 적용
output_center = np.array([output_w / 2, output_h / 2]).reshape(2, 1)    # 출력 이미지 중앙 위치

# 이동 벡터 계산
translation = output_center - center_transformed
M[:, 2] = translation.flatten()                     # 최종 행렬에 이동벡터 추가

# 최종 변환 적용
iso_map = cv2.warpAffine(map_img, M, output_size, flags=cv2.INTER_LINEAR, borderValue=(0,0,0))

# 여백 최적화
# 빨간 여백은 (0,0,255)이므로 회색값 기준으로 배경은 거의 76~85 정도
gray = cv2.cvtColor(iso_map, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # 임계값 처리로 내용만 추출
coords = cv2.findNonZero(mask)                              # 내용 있는 픽셀 좌표 추출
x, y, w_box, h_box = cv2.boundingRect(coords)               # 바운딩박스 추출
cropped = iso_map[y:y+h_box, x:x+w_box]                     # 자른 이미지 만들기

cv2.imwrite('isometric_map.png', cropped)