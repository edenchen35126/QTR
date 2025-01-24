# import cv2
# # 調整視窗大小以便顯示
# cv2.namedWindow('ORB Feature Matching', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('ORB Feature Matching', 800, 600)
# # 讀取兩張圖片
# image1_path = "D:/AI/QTR_eden/QTR/dataset/12_canny/images/output_2780.jpg"
# image2_path = "D:/AI/QTR_eden/QTR/dataset/12_canny/images/output_1120.jpg"
# image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# # 確保圖片尺寸一致
# if image1.shape != image2.shape:
#     print("圖片尺寸不同，調整為相同大小")
#     height, width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
#     image1 = cv2.resize(image1, (width, height))
#     image2 = cv2.resize(image2, (width, height))

# # 定義 ROI 區域
# # top_left = (900, 550)  # 左上角座標
# top_left = (900, 850)  # 左上角座標
# bottom_right = (2350, 1300)  # 右下角座標

# # 裁剪出 ROI 區域
# roi = image1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
# roi_2 = image2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# # 初始化 ORB 檢測器
# orb = cv2.ORB_create()

# # 提取特徵點和描述子
# keypoints1, descriptors1 = orb.detectAndCompute(roi, None)
# keypoints2, descriptors2 = orb.detectAndCompute(roi_2, None)
# print(f"圖片 1 的特徵點數量: {len(keypoints1)}")
# print(f"圖片 2 的特徵點數量: {len(keypoints2)}")

# # 使用 BFMatcher 進行特徵點匹配
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)

# # 按距離排序匹配點（距離越小越好）
# matches = sorted(matches, key=lambda x: x.distance)

# # 計算匹配點數量
# match_count = len(matches)
# print(f"匹配點數量: {match_count}")

# # 1. 基於匹配比例
# min_keypoints = min(len(keypoints1), len(keypoints2))
# match_ratio = len(matches) / min_keypoints
# ratio_threshold = 0.5  # 比例閾值

# # 2. 基於特徵豐富度
# alpha = 0.3
# dynamic_threshold = alpha * min_keypoints

# # # 3. 基於匹配距離
# # avg_distance = sum([m.distance for m in matches]) / len(matches)
# # distance_threshold = 30  # 平均距離閾值

# # # 判斷相似性
# # if match_ratio >= ratio_threshold and len(matches) >= dynamic_threshold and avg_distance < distance_threshold:
# #     print("兩張圖片相似")
# # else:
# #     print("兩張圖片不相似")

# # 判斷相似性
# if match_ratio >= ratio_threshold:
#     result = "similar"

# else:
#     result = "unsimilar"


# # # 設置相似度閾值
# # match_threshold = 50  # 需要根據實際需求調整
# # if match_count >= match_threshold:
# #     result = "similar"
# # else:
# #     result = "unsimilar"

# # 輸出結果
# print("match_ratio :",match_ratio)
# print("ratio_threshold :",ratio_threshold)
# print("result :",result)

# # 可視化匹配結果（顯示前 50 個匹配點）
# result_image = cv2.drawMatches(roi, keypoints1, roi_2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # 在圖片上顯示結果文字
# cv2.putText(result_image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# # 顯示結果
# cv2.imshow("ORB Feature Matching", result_image)
# #cv2.imwrite("D:/AI/QTR_eden/QTR/dataset/12_canny/orb_result.jpg", result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
# 調整視窗大小以便顯示
cv2.namedWindow('ORB Feature Matching', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ORB Feature Matching', 800, 600)
# 讀取圖片
image1_path = "D:/AI/QTR_eden/QTR/dataset/12_canny/images/output_2780.jpg"
image2_path = "D:/AI/QTR_eden/QTR/dataset/12_canny/images/output_740.jpg"
image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
#740
#1440 440

# 確保圖片尺寸一致
if image1.shape != image2.shape:
    print("圖片尺寸不同，調整為相同大小")
    height, width = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
    image1 = cv2.resize(image1, (width, height))
    image2 = cv2.resize(image2, (width, height))


# else:
#     image1 = cv2.resize(image1, (image1.shape[1] * 2, image1.shape[0] * 2))
#     image2 = cv2.resize(image2, (image2.shape[1] * 2, image2.shape[0] * 2))
# image1 = cv2.equalizeHist(image1)
# image2 = cv2.equalizeHist(image2)
# image1 = cv2.GaussianBlur(image1, (5, 5), 0)
# image2 = cv2.GaussianBlur(image2, (5, 5), 0)
# 定義 ROI 區域
# top_left = (900, 550)  # 左上角座標
# bottom_right = (2350, 1300)  # 右下角座標
top_left = (800, 550)  # 左上角座標
bottom_right = (2550, 1500)  # 右下角座標

# 裁剪出 ROI 區域
roi = image1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
roi_2 = image2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# 初始化 ORB 檢測器
orb = cv2.ORB_create()

# 提取特徵點和描述子
keypoints1, descriptors1 = orb.detectAndCompute(roi, None)
keypoints2, descriptors2 = orb.detectAndCompute(roi_2, None)
print(f"圖片 1 的特徵點數量: {len(keypoints1)}")
print(f"圖片 2 的特徵點數量: {len(keypoints2)}")

# 使用 BFMatcher 進行 KNN 匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 應用 Lowe's ratio test 過濾匹配點
good_matches = []
ratio_threshold = 0.85  # 比例閾值
for m, n in knn_matches:
    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)


print(f"過濾後的匹配點數量: {len(good_matches)}")

# 設置相似性閾值
match_ratio_threshold = 0.5  # 匹配比例閾值
min_keypoints = min(len(keypoints1), len(keypoints2))
match_ratio = len(good_matches) / min_keypoints


# 判斷相似性
if match_ratio >= match_ratio_threshold:
    result = "similar"
else:
    result = "unsimilar"

# if match_ratio >= match_ratio_threshold:
#     result = "similar"
# else:
#     result = "unsimilar"

# 輸出結果
print("result :",result)
print("match_ratio :",match_ratio)

print(f"匹配比例: {match_ratio:.2f}")

# 可視化匹配結果（顯示前 50 個匹配點）
result_image = cv2.drawMatches(roi, keypoints1, roi_2, keypoints2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 在結果圖片上添加判斷結果
cv2.putText(result_image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 顯示結果
cv2.imshow("ORB Feature Matching", result_image)
#cv2.imwrite("D:/AI/QTR_eden/QTR/dataset/12_canny/knn_result.jpg", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()