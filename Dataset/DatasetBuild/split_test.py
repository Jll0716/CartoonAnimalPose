import json
import os
import random
import shutil
import num_dataset

# 设置随机种子以确保结果可重复
random.seed(42)

# 原始数据集路径
annotations_dir = "../MergedAnimal_HR/annotations/"
train_annotations_file = os.path.join(annotations_dir, "train_coco.json")
val_annotations_file = os.path.join(annotations_dir, "val_coco.json")
train_images_dir = "../MergedAnimal_HR/train/"
val_images_dir = "../MergedAnimal_HR/val/"

# 输出目录
output_annotations_dir = "../MixCA/annotations/"
output_images_dir = "../MixCA/"

# 创建输出目录
os.makedirs(output_annotations_dir, exist_ok=True)
os.makedirs(os.path.join(output_images_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_images_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_images_dir, "test"), exist_ok=True)

# 加载原始注释文件
with open(train_annotations_file, "r") as f:
    train_annotations = json.load(f)

with open(val_annotations_file, "r") as f:
    val_annotations = json.load(f)

# 合并 train 和 val 的图片及注释信息
images = train_annotations["images"] + val_annotations["images"]
annotations_data = train_annotations["annotations"] + val_annotations["annotations"]
categories = train_annotations["categories"]  # 假设两者的类别信息一致

# 打乱数据集
random.shuffle(images)

# 按照 7:2:1 划分数据集
total_count = len(images)
train_count = int(0.7 * total_count)
val_count = int(0.2 * total_count)

train_images_split = images[:train_count]
val_images_split = images[train_count:train_count + val_count]
test_images_split = images[train_count + val_count:]

# 创建字典 {image_id: image} 方便查找
image_id_to_image = {img["id"]: img for img in images}

# 根据图片划分注释
def split_annotations(images_split):
    image_ids = {img["id"] for img in images_split}
    split_annotations = [ann for ann in annotations_data if ann["image_id"] in image_ids]
    return images_split, split_annotations

train_images_final, train_annotations_final = split_annotations(train_images_split)
val_images_final, val_annotations_final = split_annotations(val_images_split)
test_images_final, test_annotations_final = split_annotations(test_images_split)

# 保存新的注释文件
def save_annotations(images, annotations, filename):
    data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,  # 保留类别信息
    }
    with open(os.path.join(output_annotations_dir, filename), "w") as f:
        json.dump(data, f)

save_annotations(train_images_final, train_annotations_final, "train_coco.json")
save_annotations(val_images_final, val_annotations_final, "val_coco.json")
save_annotations(test_images_final, test_annotations_final, "test_coco.json")

# 复制图片到对应文件夹
def copy_images(images_split, source_dirs, dest_dir):
    for img in images_split:
        src_path = None
        for source_dir in source_dirs:
            potential_path = os.path.join(source_dir, img["file_name"])
            if os.path.exists(potential_path):
                src_path = potential_path
                break
        if src_path:
            shutil.copy(src_path, os.path.join(dest_dir, img["file_name"]))

copy_images(train_images_final, [train_images_dir, val_images_dir], os.path.join(output_images_dir, "train"))
copy_images(val_images_final, [train_images_dir, val_images_dir], os.path.join(output_images_dir, "val"))
copy_images(test_images_final, [train_images_dir, val_images_dir], os.path.join(output_images_dir, "test"))

print("数据集划分完成，新文件已生成：")
print(f"训练集注释：{os.path.join(output_annotations_dir, 'train_coco.json')}")
print(f"验证集注释：{os.path.join(output_annotations_dir, 'val_coco.json')}")
print(f"测试集注释：{os.path.join(output_annotations_dir, 'test_coco.json')}")
print(f"训练集图片目录：{os.path.join(output_images_dir, 'train/')}")
print(f"验证集图片目录：{os.path.join(output_images_dir, 'val/')}")
print(f"测试集图片目录：{os.path.join(output_images_dir, 'test/')}")

num_dataset.main()