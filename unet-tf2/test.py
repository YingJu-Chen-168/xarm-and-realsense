# # yolo的一些參數
# boxes = results[0].boxes.xyxy.tolist()
# classes = results[0].boxes.cls.tolist()
# names = results[0].names
# confidences = results[0].boxes.conf.tolist()
# masks = results[0].masks

# 引入u_net、相片
from unet_tf2.models.unet import Unet
from unet_tf2.utils import dice_coef, iou_coef
from keras.metrics import Precision, Recall
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

features =  24
levels =  5
epochs = 10
batch_size =  8
learning_rate = 9e-4 
seed = 2202
SIZE = 320
optimizer = tf.keras.optimizers.Adam(learning_rate)
model = Unet((SIZE, SIZE, 3),  features, levels)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[ dice_coef, iou_coef, Precision(), Recall()])
model.load_weights("my_model.h5", by_name=True)
# model.summary()
image = cv2.imread('100.jpg')
image = np.asarray(image)
image = np.float32(np.array(image) / 255)
image = tf.image.resize(image, (SIZE, SIZE))
image = tf.expand_dims(image, 0)
# 預測
y_pred = model.predict(image)
y_pred = tf.image.resize(y_pred, [320, 320] )
y_pred = y_pred.numpy()
y_pred = 1*(y_pred > 0.5)
mask_img = np.uint8( y_pred[0]*255 )
color_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
# y_pred = y_pred[:, :, 0]
ground_img = cv2.imread('100.png')

diff_mask = np.any(color_img != ground_img, axis=-1)    # axis=-1 代表最後一個維度(BGR通道)
same_mask = np.any(color_img == ground_img, axis=-1)
ground_foreground = np.all(ground_img == [255, 255, 255], axis=-1)  # Ground truth 前景
ground_background = np.all(ground_img == [0, 0, 0], axis=-1)        # Ground truth 背景
pred_foreground = np.all(color_img == [255, 255, 255], axis=-1)  # 預測為前景
pred_background = np.all(color_img == [0, 0, 0], axis=-1)        # 預測為背景

color_img[np.logical_and(diff_mask, pred_background)] = [0, 0, 255]      # 紅色 FN
color_img[np.logical_and(diff_mask, pred_foreground)] = [255, 105, 65]   # 藍色 FP
color_img[np.logical_and(same_mask, pred_foreground)] = [70, 190, 100]
# error = np.any(color_img != ground_img, axis=-1)  # 找出所有不相等的像素 (True/False)
# color_img[error] = [0, 0, 255]

# 計算混淆矩陣指標
TP = np.sum(np.logical_and(pred_foreground, ground_foreground))  # 真的偵測到前景
TN = np.sum(np.logical_and(pred_background, ground_background))  # 真的偵測到背景
FP = np.sum(np.logical_and(pred_foreground, ground_background))  # 誤判為前景
FN = np.sum(np.logical_and(pred_background, ground_foreground))  # 漏掉的前景

iou = TP/(TP+FN+FP)
dice = (2*TP)/(2*TP+FN+FP)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

cv2.imshow("result", color_img)
cv2.waitKey(0)
cv2.imwrite("result.jpg", color_img)

# 印出結果
print(f"iou : {iou}")
print(f"dice : {dice}")
print(f"precision : {precision}")
print(f"recall : {recall}")

# plt.imshow(color_img, cmap = "gray")
# plt.show()

# # 檢查h5py檔案裡面的內容
# import h5py
# with h5py.File('temp_run=0_weights.h5', 'r') as f:
#     print(f.keys()) # <KeysViewHDF5 ['layers', 'optimizer', 'vars']> # layers : 保存了模型中每一層的權重資訊
#     # optimizer : 包含優化器的狀態資訊，例如動量、學習率調整相關數據 vars : 一般用來保存與模型變數相關的元數據