import os
import json
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
# 验证COCO
# my_coco = COCO('test.json')

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


def tococo(input, output):
    with open(input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # print(data)
    # data = animalpose
    coco = {}
    coco['categories'] = []
    coco['categories'].append(class_list)
    coco['images'] = []
    coco['annotations'] = []

# ## 提取图像元数据
    #                 img_dict = {}
    #                 img_dict['file_name'] = labelme['imagePath']
    #                 img_dict['height'] = labelme['imageHeight']
    #                 img_dict['width'] = labelme['imageWidth']
    #                 img_dict['id'] = IMG_ID
    #                 coco['images'].append(img_dict)
    ANN_ID = 1

    for annotations in data['annotations']:
        img_id = annotations['image_id']
        image_bbox = annotations['bbox']
        img_file = data['images'][str(img_id)]
        xmin, ymin, xmax, ymax = image_bbox
        image_bbox = [xmin, ymin, xmax-xmin, ymax-ymin]

        img_folder = 'images/'
        img_path = os.path.join(img_folder,img_file)
        with Image.open(img_path) as img:
            width, height = img.size
        image_dict = {"file_name": img_file, "height": height, "width": width, "id": img_id}
        if image_dict not in coco['images']:
            coco['images'].append(image_dict)

        keypoints = annotations['keypoints']
        img_keypoints = []
        num_keypoints = 0
        for k in keypoints:
            if k[2]==1:
                num_keypoints+=1
            img_keypoints.extend(k)
        img_keypoints.extend([0,0,0])
        area = image_bbox[2] * image_bbox[3]
        annotation_dict = {"category_id":1,"iscrowd":0,"image_id":img_id,"id":ANN_ID,"bbox":image_bbox,
                           "num_keypoints":21, "area":area, "keypoints":img_keypoints}
        coco['annotations'].append(annotation_dict)
        ANN_ID+=1

    with open(output,'w') as f:
        json.dump(coco, f, indent=4)


if __name__ == "__main__":
    input = 'keypoints.json'
    output = 'animalpose_coco.json'
    tococo(input,output)
    my_coco = COCO(output)

# print(apjson.keys())
# print(apjson['images'])