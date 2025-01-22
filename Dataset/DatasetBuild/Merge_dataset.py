import os
import json


def load_coco_json(json_path):
    """加载 COCO 格式的 JSON 文件"""
    with open(json_path, 'r') as f:
        return json.load(f)


def merge_coco_annotations(coco1, coco2):
    """合并两个 COCO 格式的注释"""
    merged_coco = {
        "images": [],
        "annotations": [],
        "categories": coco1["categories"]  # 使用第一个数据集的 categories
    }

    # ID 偏移量
    image_id_offset = len(coco1['images'])
    annotation_id_offset = len(coco1['annotations'])

    # 合并第一个数据集
    merged_coco['images'].extend(coco1['images'])
    merged_coco['annotations'].extend(coco1['annotations'])

    # 合并第二个数据集
    for image in coco2['images']:
        image['id'] += image_id_offset  # 更新 image_id
        merged_coco['images'].append(image)

    for annotation in coco2['annotations']:
        annotation['id'] += annotation_id_offset  # 更新 annotation_id
        annotation['image_id'] += image_id_offset  # 更新 image_id
        merged_coco['annotations'].append(annotation)

    return merged_coco


def save_coco_json(coco_data, output_path):
    """保存 COCO 格式的 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


def main():
    # 第一个数据集注释文件
    dataset1_train_json = '../CAP4000/annotations/train_coco.json'
    dataset1_val_json = '../CAP4000/annotations/val_coco.json'

    # 第二个数据集注释文件
    dataset2_train_json = '../AnimalPose_HR/annotations/train_coco.json'
    dataset2_val_json = '../AnimalPose_HR/annotations/val_coco.json'

    # 合并后的注释文件保存路径
    output_train_json = '../MixCA/annotations/train_coco.json'
    output_val_json = '../MixCA/annotations/val_coco.json'

    # 加载两个数据集的注释
    coco1_train = load_coco_json(dataset1_train_json)
    coco1_val = load_coco_json(dataset1_val_json)
    coco2_train = load_coco_json(dataset2_train_json)
    coco2_val = load_coco_json(dataset2_val_json)

    # 合并 train 和 val 注释
    merged_train_coco = merge_coco_annotations(coco1_train, coco2_train)
    merged_val_coco = merge_coco_annotations(coco1_val, coco2_val)

    # 保存合并后的注释
    save_coco_json(merged_train_coco, output_train_json)
    save_coco_json(merged_val_coco, output_val_json)
    print(f"Number of train in annotations: {len(merged_train_coco['images'])}")
    print(f"Number of val in annotations: {len(merged_val_coco['images'])}")
    print(f"Train annotations saved to {output_train_json}")
    print(f"Val annotations saved to {output_val_json}")


if __name__ == "__main__":
    main()
