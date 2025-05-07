import cv2
import numpy as np
import glob,os

UNKNOWN_AREA_MASK = (135,204,207) # (R,G,B)
EXCEPT_PIXEL_RANGE = 3
GRID_PATH = "./grid_map/"
MAINMAP_PATH = "./main_map/"
CROP_PATH = "./cropped_map/"
ISOMATRIC_PATH = "./isomatric_map/"
BLENDED_PATH = "./blended_map/"

grid = glob.glob(GRID_PATH+"*.png")
main_mapz = glob.glob(MAINMAP_PATH+"*.png")

def crop(path):
    img = cv2.imread(path)

    # copy
    copy = img.copy()

    # except pixel crop
    pure = copy[EXCEPT_PIXEL_RANGE:img.shape[0] - EXCEPT_PIXEL_RANGE,
                EXCEPT_PIXEL_RANGE:img.shape[1] - EXCEPT_PIXEL_RANGE]

    # bgr channel split
    b, g, r = cv2.split(pure)

    # mask
    mask = (r == 135) & (g == 204) & (b == 207)
    pure[mask] = [0, 0, 0]

    gray = cv2.cvtColor(pure, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) 

    coords = cv2.findNonZero(mask)

    # bbox
    crop_range = cv2.boundingRect(coords)

    # crop
    x, y, w, h = crop_range
    cropped_img = pure[y:y+h,
                        x:x+w]
    # cropped_img = img[y+EXCEPT_PIXEL_RANGE:y+h+EXCEPT_PIXEL_RANGE,
    #                 x+EXCEPT_PIXEL_RANGE:x+w+EXCEPT_PIXEL_RANGE]
    

    return cropped_img

def isomatric_trancefrom(img):
    h, w = img.shape[:2]

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
    iso_map = cv2.warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR, borderValue=(0,0,0))

    # 여백 최적화
    gray = cv2.cvtColor(iso_map, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # 임계값 처리로 내용만 추출

    coords = cv2.findNonZero(mask)                              # 내용 있는 픽셀 좌표 추출
    x, y, w_box, h_box = cv2.boundingRect(coords)               # 바운딩박스 추출

    cropped = iso_map[y:y+h_box, x:x+w_box]                     # 자른 이미지 만들기
    return cropped

def alpha_blend(grid,main,alpha=0.3,beta=0.7): # 실패 ;;;
    grid = cv2.cvtColor(grid,cv2.COLOR_BGR2BGRA)
    main = cv2.cvtColor(main,cv2.COLOR_BGR2BGRA)

    alpha = 0.3
    beta = 0.7

    h,w,c = main.shape
    #####test######
    grid_map_ratio = grid.shape[0]/grid.shape[1]
    new_height = int(w * grid_map_ratio)
    grid_resized = cv2.resize(grid, (w, new_height))
    height_diff = abs(h - new_height)
    pad_top = height_diff // 2
    pad_bottom = height_diff - pad_top

    # 세로 길이가 작은 경우: 패딩 추가
    if new_height < h:
        grid_padded = cv2.copyMakeBorder(grid_resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        blended_img = cv2.addWeighted(main, alpha, grid_padded, beta, 0)

    # 세로 길이가 큰 경우: 메인 맵에 패딩 추가
    elif new_height > h:
        icogram_padded = cv2.copyMakeBorder(main, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        blended_img = cv2.addWeighted(icogram_padded, alpha, grid_resized, beta, 0)

    # 세로 길이가 같은 경우: 바로 블렌딩
    else:
        blended_img = cv2.addWeighted(main, alpha, grid_resized, beta, 0)

    return blended_img
    ###############
    grid = cv2.resize(grid,(w,h))

    blended_img = (main * alpha + grid * beta).astype(np.uint8)
    return blended_img

def main_process():
    for grid_path in grid:
        basename = os.path.basename(grid_path)
        name,ext = os.path.splitext(basename)

        # preprocess
        crop_img = crop(grid_path)
        new_path = CROP_PATH + name + "_cropped" + ext
        cv2.imwrite(new_path,crop_img)

        # tranceform isomatric_view
        isomatric = isomatric_trancefrom(crop_img)
        new_path = ISOMATRIC_PATH + name + "_isomatric" + ext
        cv2.imwrite(new_path,isomatric)


        #### 안됨 ####
        for mainmap_path in main_mapz:
            if "2F" in name and "2F" in mainmap_path:
                main_map = cv2.imread(mainmap_path)
                print(f"2층 그리드맵 = {isomatric.shape}")
                print(f"2층 그리드맵 세로/가로 = {isomatric.shape[0]/isomatric.shape[1]}")
                print(f"2층 메인맵 = {main_map.shape}")
                print(f"2층 메인맵 세로/가로 = {main_map.shape[0]/main_map.shape[1]}")
                blended = alpha_blend(grid=isomatric,main=main_map)
                new_path = BLENDED_PATH + name + "_blended" + ext
                cv2.imwrite(new_path,blended)

            if "3F" in name and "3F" in mainmap_path:
                main_map = cv2.imread(mainmap_path)
                print(f"3층 그리드맵 = {isomatric.shape}")
                print(f"3층 그리드맵 세로/가로 = {isomatric.shape[0]/isomatric.shape[1]}")
                print(f"3층 메인맵 = {main_map.shape}")
                print(f"3층 메인맵 세로/가로 = {main_map.shape[0]/main_map.shape[1]}")
                blended = alpha_blend(grid=isomatric,main=main_map)
                new_path = BLENDED_PATH + name + "_blended" + ext
                cv2.imwrite(new_path,blended)

            else:
                blended = None
        ###############

