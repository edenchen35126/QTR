import cv2
import os



def similar_compare(image1,roi,roi_2,num):
    # 初始化對齊狀態
    aligned = True

    # 檢查每個點的像素值變化
    for (x, y) in points:
        if y >= roi.shape[0] or x >= roi.shape[1]:  # 確保點在 ROI 範圍內
            print(f"點 ({x}, {y}) 超出範圍，請檢查 ROI 設定")
            continue

        # 獲取兩張圖片中該點的像素值
        pixel_value_1 = roi[y, x]
        pixel_value_2 = roi_2[y, x]

        # 計算像素值的差異
        diff = cv2.norm(pixel_value_1, pixel_value_2, cv2.NORM_L2)

        # 計算像素值的差異
        #diff = abs(int(pixel_value_1) - int(pixel_value_2))

        if diff > threshold:
            aligned = False
            print(f"點 ({x}, {y}) 的像素值不同：{pixel_value_1} vs {pixel_value_2}, 差異 {diff}")
            cv2.circle(roi_2, (x, y), 5, (0, 0, 255), -1)  # 用紅色標記未對齊的點
        else:
            #print(f"點 ({x}, {y}) 的像素值相同：{pixel_value_1} vs {pixel_value_2}")
            cv2.circle(roi_2, (x, y), 5, (0, 255, 0), -1)  # 用綠色標記對齊的點

    # 將處理後的 ROI 放回原圖
    image2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi_2

    # 繪製 ROI 區域邊框
    cv2.rectangle(image2, top_left, bottom_right, (255, 255, 0), 2)  # 藍色框標記 ROI 區域

    if aligned:
        print("aligned")
        line.append(f"{num} is aligned \n")
    else:
        print("not aligned")
        line.append(f"{num} is not aligned \n")
    # 調整視窗大小以便顯示
    cv2.namedWindow('Pixel Value Comparison', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pixel Value Comparison', 800, 600)
    # 顯示結果
    cv2.imshow("Pixel Value Comparison", image2)
    #cv2.imwrite("D:/AI/QTR_eden/QTR/dataset/12_canny/result_pixel_comparison.jpg", image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 讀取兩張圖片
image_path = "D:/AI/QTR_eden/QTR/dataset/12_canny/images/output_2780.jpg"

# 定義特定點座標 (相對於 ROI 的座標)
points = [(100, 70),(400,60),(550,50),(850,35),(1150,25),(1300,20),(715,150),(725,250),(730,350)]  # 替換為您感興趣的點
#100, 80

line = []

# 定義閾值
threshold = 40 # 如果像素值變化超過此值，則視為不一致

test_Path = "D:/AI/QTR_eden/QTR/dataset/12_canny/unsimilar/"
testFilePath = os.listdir(test_Path)
test_record = []
for file in testFilePath:
    test_record.append(file)

for i in range(len(test_record)):
    imagepath = ""+test_Path+test_record[i]+""
    image1 = cv2.imread(image_path)
    image2 = cv2.imread(imagepath)

    top_left = (900, 550)  # 左上角座標
    bottom_right = (2350, 1500)  # 右下角座標

    # 裁剪出 ROI 區域
    roi = image1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    roi_2 = image2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    similar_compare(image1,roi,roi_2,test_record[i])


# 開啟檔案並寫入多行
with open("D:/AI/QTR_eden/QTR/dataset/12_canny/record.txt", "w", encoding="utf-8") as file:
    file.writelines(line)




# image_path_2 = "D:/AI/QTR_eden/QTR/dataset/12_canny/images/output_460.jpg"
# image = cv2.imread(image_path)
# image_2 = cv2.imread(image_path_2)
# #740
# #1440

# # 定義 ROI 區域
# top_left = (900, 550)  # 左上角座標
# bottom_right = (2350, 1500)  # 右下角座標

# # 裁剪出 ROI 區域
# roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
# roi_2 = image_2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# # 定義特定點座標 (相對於 ROI 的座標)
# points = [(100, 80),(400,60),(550,50),(850,35),(1150,25),(1300,20),(715,150),(725,250),(730,350)]  # 替換為您感興趣的點
# # (12,280),(15,480),(30,680),左邊沒有紅外線

# # 定義閾值
# threshold = 30  # 如果像素值變化超過此值，則視為不一致

