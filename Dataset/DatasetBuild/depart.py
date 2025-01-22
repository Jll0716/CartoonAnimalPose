# 用于划分注释
# 划分图片
import json
import random
from pycocotools.coco import COCO
import shutil
import os

def load_coco_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def split_coco_annotations(coco_data, train_ratio=0.8):
    random.seed(42)  # 为了确保可重现性
    images = coco_data['images']
    annotations = coco_data['annotations']
    # 计算分割数量
    num_train = int(len(images) * train_ratio)
    # 随机选择图像
    random.shuffle(images)
    train_images = images[:num_train]
    val_images = images[num_train:]
    # 为训练集和验证集创建新的注释结构
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    # 创建新的COCO格式数据集
    train_coco = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories'],
    }
    val_coco = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories'],
    }
    return train_coco, val_coco


def save_coco_json(coco_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


def copy_images(coco_data, src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    image_ids = {img['id']: img['file_name'] for img in coco_data['images']}

    for img_id, file_name in image_ids.items():
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f'Copied {file_name} to {dst_folder}')
        else:
            print(f'Warning: {file_name} does not exist in {src_folder}')


# 划分注释
def depart_ann():
    input_annotations_path = 'animalpose_coco.json'  # 输入的COCO格式注释文件路径
    train_output_path = 'train_animalpose_coco.json'  # 输出的训练集注释文件路径
    val_output_path = 'val_animalpose_coco.json'  # 输出的验证集注释文件路径
    # 加载原始COCO注释
    coco_data = load_coco_json(input_annotations_path)
    # 分割数据集
    train_coco, val_coco = split_coco_annotations(coco_data)
    # 保存分割后的训练集和验证集注释
    save_coco_json(train_coco, train_output_path)
    save_coco_json(val_coco, val_output_path)
    print(f'Train annotations saved to {train_output_path}')
    print(f'Validation annotations saved to {val_output_path}')


# 划分图片
def depart_img():
    train_annotations_path = '../MergedAnimal_HR/annotations/train_coco.json'  # 训练集注释路径
    val_annotations_path = '../MergedAnimal_HR/annotations/val_coco.json'  # 验证集注释路径
    images_folder = '../MergedAnimal_HR/image'  # 原始图像文件夹路径
    train_images_folder = '../MergedAnimal_HR//train'  # 目标训练集图像文件夹
    val_images_folder = '../MergedAnimal_HR//val'  # 目标验证集图像文件夹

    # 加载注释文件
    train_coco = load_coco_json(train_annotations_path)
    val_coco = load_coco_json(val_annotations_path)
    # 复制训练集图像
    copy_images(train_coco, images_folder, train_images_folder)
    # 复制验证集图像
    copy_images(val_coco, images_folder, val_images_folder)
    print('Image copying completed.')


if __name__ == "__main__":
    # depart_ann()
    depart_img()
    # my_coco = COCO('val_animalpose_coco.json')