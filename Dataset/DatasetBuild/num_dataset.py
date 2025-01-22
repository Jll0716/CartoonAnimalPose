import json
import os

def load_coco_annotations(file_path):
    """加载 COCO 格式的 annotations 文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def count_images_and_annotations(annotations_file):
    """统计 COCO 格式数据集的图片和注释数量"""
    data = load_coco_annotations(annotations_file)
    num_images = len(data.get('images', []))
    num_annotations = len(data.get('annotations', []))
    return num_images, num_annotations

def main():
    # 修改为你的数据集路径
    dataset_path = "../AnimalPose_HR/annotations"
    train_file = os.path.join(dataset_path, "train_coco.json")
    val_file = os.path.join(dataset_path, "val_coco.json")
    # test_file = os.path.join(dataset_path, "test_coco.json")

    # 检查文件是否存在
    if not os.path.exists(train_file):
        print(f"训练集文件不存在: {train_file}")
        return

    if not os.path.exists(val_file):
        print(f"验证集文件不存在: {val_file}")
        return

    # if not os.path.exists(test_file):
    #     print(f"验证集文件不存在: {test_file}")
    #     return

    # 统计训练集和验证集的图片和注释数量
    train_images, train_annotations = count_images_and_annotations(train_file)
    val_images, val_annotations = count_images_and_annotations(val_file)
    #test_images, test_annotations = count_images_and_annotations(test_file)

    # 打印统计结果
    print(f"训练集: 图片数量 = {train_images}, 注释数量 = {train_annotations}")
    print(f"验证集: 图片数量 = {val_images}, 注释数量 = {val_annotations}")
    #print(f"测试集: 图片数量 = {test_images}, 注释数量 = {test_annotations}")
    print(f"数据集: 图片数量 = {train_images + val_images }, 注释数量 = {train_annotations + val_annotations }")
    #print(f"数据集: 图片数量 = {train_images+val_images+test_images}, 注释数量 = {train_annotations+val_annotations+test_annotations}")


if __name__ == "__main__":
    main()
