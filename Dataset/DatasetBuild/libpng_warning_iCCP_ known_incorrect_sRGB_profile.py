import os
from tqdm import tqdm
import cv2
from skimage import io
train_path = r"/home/jll/Pictures/img_1/"
val_path =  r"/home/jll/Pictures/img_1/"
fileList = os.listdir(train_path)
for i in tqdm(fileList):
    image = io.imread(train_path+i)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imencode('.png', image)[1].tofile(train_path+i)
    # print(f"训练集：{i}.png处理完毕")

fileList = os.listdir(val_path)
for i in tqdm(fileList):
    image = io.imread(val_path+i)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imencode('.png', image)[1].tofile(val_path+i)
    # print(f"测试集：{i}.png处理完毕")


# GPT版本
'''
import os
from tqdm import tqdm
import cv2
from skimage import io

train_path = r"CartoonAnimal1500_HR/train/"
val_path = r"CartoonAnimal1500_HR/val/"

# 处理训练集图像
fileList = os.listdir(train_path)
for i in tqdm(fileList):
    if i.endswith(".png"):  # 确保是PNG文件
        image = io.imread(train_path + i)
        cv2.imencode('.png', image)[1].tofile(train_path + i)

# 处理验证集图像
fileList = os.listdir(val_path)
for i in tqdm(fileList):
    if i.endswith(".png"):  # 确保是PNG文件
        image = io.imread(val_path + i)
        cv2.imencode('.png', image)[1].tofile(val_path + i)

'''
