# 原数据集格式：labelmejson & images
# 首先划分训练集和测试集，仅需要对images进行划分

import os
import shutil
import random
import filecmp
import json
import numpy as np
import seedir as sd
from tqdm import tqdm
from pycocotools.coco import COCO

ANN_ID = 0
IMG_ID = 0

def compare_directories(source_dir, target_dir):
    # 检查源文件夹和目标文件夹内容是否一致
    dirs_cmp = filecmp.dircmp(source_dir, target_dir)
    if not dirs_cmp.diff_files and not dirs_cmp.left_only and not dirs_cmp.right_only:
        print("两文件夹下的内容一致。不需要进行复制。")
        return False
    return True


# 复制数据集，保持源文件不损坏
# def copy_dataset():
def copy_dataset(source_dir, target_dir):
    if compare_directories(source_dir, target_dir):
        # 复制目录下的所有文件和子目录到目标路径
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(target_dir, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, target_item)
            else:
                shutil.copy2(source_item, target_item)
        print("复制结束")


# 查看该数据集的文件数目
def check_dataset(dir):
    os.chdir(os.path.join(dir , 'labelme_jsons'))
    print(f"在{os.getcwd()}目录下共有 {len(os.listdir())} 个 labelme 格式的 json 文件")


# 划分训练集、测试集
def divide(dir, test_frac):
    os.chdir(os.path.join(dir, 'labelme_jsons'))

    test_frac = test_frac  # 测试集比例
    random.seed(123)  # 随机数种子，便于复现

    folder = '.'
    img_paths = os.listdir(folder)
    random.shuffle(img_paths)  # 随机打乱

    val_number = int(len(img_paths) * test_frac)  # 测试集文件个数
    train_files = img_paths[val_number:]  # 训练集文件名列表
    val_files = img_paths[:val_number]  # 测试集文件名列表
    print('数据集文件总数', len(img_paths))
    print('训练集文件个数', len(train_files))
    print('测试集文件个数', len(val_files))

    # 创建文件夹，存放训练集的 labelme格式的 json 标注文件
    train_labelme_jsons_folder = 'train_labelme_jsons'
    os.mkdir(train_labelme_jsons_folder)
    for each in tqdm(train_files):
        src_path = os.path.join(folder, each)
        dst_path = os.path.join(train_labelme_jsons_folder, each)
        shutil.move(src_path, dst_path)
    # 创建文件夹，存放训练集的 labelme格式的 json 标注文件
    val_labelme_jsons_folder = 'val_labelme_jsons'
    os.mkdir(val_labelme_jsons_folder)
    for each in tqdm(val_files):
        src_path = os.path.join(folder, each)
        dst_path = os.path.join(val_labelme_jsons_folder, each)
        shutil.move(src_path, dst_path)
    # 检查文件总数是否不变
    print(f"训练集和测试集总数为：{len(os.listdir(train_labelme_jsons_folder)) + len(os.listdir(val_labelme_jsons_folder))}")


# 单个json文件转换
def process_single_json(labelme, class_list, image_id=1):
    '''
    输入labelme的json数据，输出coco格式的每个框的关键点标注信息
    '''
    global ANN_ID
    coco_annotations = []
    for each_ann in labelme['shapes']:  # 遍历该json文件中的所有标注
        if each_ann['shape_type'] == 'rectangle':  # 筛选出个体框
            # 个体框元数据
            bbox_dict = {}
            bbox_dict['category_id'] = 1
            bbox_dict['segmentation'] = []

            bbox_dict['iscrowd'] = 0
            bbox_dict['segmentation'] = []
            bbox_dict['image_id'] = image_id
            bbox_dict['id'] = ANN_ID
            # print(ANN_ID)
            ANN_ID += 1

            # 获取个体框坐标
            bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_w = bbox_right_bottom_x - bbox_left_top_x
            bbox_h = bbox_right_bottom_y - bbox_left_top_y
            bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]  # 左上角x、y、框的w、h
            bbox_dict['area'] = bbox_w * bbox_h

            # 筛选出分割多段线
            for each_ann in labelme['shapes']:  # 遍历所有标注
                if each_ann['shape_type'] == 'polygon':  # 筛选出分割多段线标注
                    # 第一个点的坐标
                    first_x = each_ann['points'][0][0]
                    first_y = each_ann['points'][0][1]
                    if (first_x > bbox_left_top_x) & (first_x < bbox_right_bottom_x) & (
                            first_y < bbox_right_bottom_y) & (first_y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                        bbox_dict['segmentation'] = list(
                            map(lambda x: list(map(lambda y: round(y, 2), x)), each_ann['points']))  # 坐标保留两位小数
                        # bbox_dict['segmentation'] = each_ann['points']

            # 筛选出该个体框中的所有关键点
            bbox_keypoints_dict = {}
            for each_ann in labelme['shapes']:  # 遍历所有标注

                if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                    # 关键点横纵坐标
                    x = int(each_ann['points'][0][0])
                    y = int(each_ann['points'][0][1])
                    label = each_ann['label']
                    if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & (y < bbox_right_bottom_y) & (
                            y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                        bbox_keypoints_dict[label] = [x, y]

            bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
            # print(bbox_keypoints_dict)

            # 把关键点按照类别顺序排好
            bbox_dict['keypoints'] = []
            for each_class in class_list['keypoints']:
                if each_class in bbox_keypoints_dict:
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0])
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1])
                    bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
                else:  # 不存在的点，一律为0
                    bbox_dict['keypoints'].append(0)
                    bbox_dict['keypoints'].append(0)
                    bbox_dict['keypoints'].append(0)

            coco_annotations.append(bbox_dict)

    return coco_annotations


# 当前目录下的json文件转换
def process_folder(coco, class_list):
    IMG_ID = 0
    ANN_ID = 0
    # 遍历所有 labelme 格式的 json 文件
    for labelme_json in os.listdir():

        if labelme_json.split('.')[-1] == 'json':

            with open(labelme_json, 'r', encoding='utf-8') as f:

                labelme = json.load(f)
                IMG_ID += 1
                #
                print('正在处理', labelme_json)
                ## 提取图像元数据
                img_dict = {}
                img_dict['file_name'] = labelme['imagePath']
                img_dict['height'] = labelme['imageHeight']
                img_dict['width'] = labelme['imageWidth']
                img_dict['id'] = IMG_ID
                coco['images'].append(img_dict)

                ## 提取框和关键点信息
                coco_annotations = process_single_json(labelme, class_list=class_list, image_id=IMG_ID)
                coco['annotations'] += coco_annotations

                # IMG_ID += 1
                #
                # print(labelme_json, '已处理完毕')

        else:
            pass


# 批量进行格式转换
def to_coco(data_root):
    # 指定数据集信息
    class_list = {
        'supercategory': 'animal',
        'id': 1,
        'name': 'tetrapod',
        # 大小写敏感
        'keypoints': ['L_eye', 'R_eye', 'nose', 'L_ear', 'R_ear', 'L_F_elbow', 'R_F_elbow',
                      'L_B_elbow', 'R_B_elbow', 'L_F_knee', 'R_F_knee', 'L_B_knee', 'R_B_knee',
                      'L_F_paw', 'R_F_paw', 'L_B_paw', 'R_B_paw', 'throat', 'withers', 'tail', 'torso'],
        'skeleton': [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 17], [5, 9], [5, 20], [6, 10], [6, 20], [7, 11],
                     [7, 20], [8, 12], [8, 20], [9, 13], [10, 14], [11, 15], [12, 16], [18, 19]]
    }

    # 转换训练集
    train_coco = {}
    train_coco['categories'] = []
    train_coco['categories'].append(class_list)
    train_coco['images'] = []
    train_coco['annotations'] = []
    global IMG_ID
    global ANN_ID
    IMG_ID = 0
    ANN_ID = 0

    path = os.path.join(data_root, 'labelme_jsons', 'train_labelme_jsons')
    os.chdir(path)
    process_folder(train_coco, class_list)
    # 保存coco标注文件
    coco_path = '../../train_coco.json'
    with open(coco_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    # 验证训练集
    os.chdir('../../../')
    my_coco = COCO('train_coco.json')
    print(my_coco)
    if (my_coco):
        print("训练集转换完毕。")
    else:
        raise ValueError(f"训练集转换失败")



    # 转换测试集
    val_coco = {}
    val_coco['categories'] = []
    val_coco['categories'].append(class_list)
    val_coco['images'] = []
    val_coco['annotations'] = []
    IMG_ID = 0
    ANN_ID = 0
    path = os.path.join('labelme_jsons', 'val_labelme_jsons')
    os.chdir(path)
    process_folder(val_coco, class_list)
    # 保存coco标注文件
    coco_path = '../../val_coco.json'
    with open(coco_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    os.chdir('../../../')
    #验证测试集
    my_coco = COCO('val_coco.json')
    print(my_coco)
    if (my_coco):
        print("测试集转换完毕。")
    else:
        raise ValueError(f"测试集转换失败")

    # 删除原labelme的json文件目录
    directory = "labelme_jsons"
    if os.path.exists(directory):
        # 递归地删除目录
        shutil.rmtree(directory)
        print(f"已删除目录：{directory}。")
    else:
        print(f"待删除的目录：'{directory}' 不存在。")


# 比较两文件大小是否一致
def compare_files(file1_path, file2_path):
    with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
        content1 = file1.read()
        content2 = file2.read()

    if content1 == content2:
        print(f"The contents of {file1_path} and {file2_path} are the same.")
    else:
        print(f"The contents of {file1_path} and {file2_path} are different.")


if __name__ == '__main__':
    # 原数据集目录
    data_origin_root = '/home/jll/Videos/AnimalSimpleArt'
    if os.path.exists(data_origin_root):
        print("Path Origin exists")
    else:
        raise FileNotFoundError(f"Path does not exist: {data_origin_root}")

    # 格式修改后的数据集目录
    data_trans_root = '/home/jll/Videos/CartoonAnimal_RTM'
    if os.path.exists(data_trans_root):
        print("Path Target exists")
    else:
        os.makedirs(data_trans_root)
        print(f"Created directory: {data_trans_root}")

    # 复制原数据集到目标数据集
    copy_dataset(data_origin_root, data_trans_root)
    # 检查复制后的数据集和复制前的数据集大小是否相同
    check_dataset(data_origin_root)
    check_dataset(data_trans_root)

    # 划分测试集和训练集
    divide(data_trans_root, test_frac=0.2)

    # 格式转换
    to_coco(data_trans_root)

    # 查看转换完成后的数据集目录
    sd.seedir(data_trans_root, style='emoji', depthlimit=1)

    # print("Turn " + dataroot +" to coco for rtmpose succeed!")
    # file1 = "/home/jll/Videos/CartoonAnimal_RTM/train_coco.json"
    # file2 = "/home/jll/Anaconda/Jupyter/SeeMyLabel/Label2Coco/AnimalSimpleArt/train_coco.json"
    # compare_files(file1, file2)