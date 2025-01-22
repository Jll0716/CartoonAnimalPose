"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import HighResolutionNet
from model.ahrnet import HighResolutionNet_Attention
from model.csa_net import HighResolutionNet_CSA
from model.cbam import HighResolutionNet_CBAM
from train_utils import EvalCOCOMetric
from my_dataset_coco import CocoKeypoint
import transforms
import time


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 10, [""] * 10
    stats[0], print_list[0] = _summarize(1, maxDets=20)
    stats[1], print_list[1] = _summarize(1, maxDets=20, iouThr=.5)
    stats[2], print_list[2] = _summarize(1, maxDets=20, iouThr=.75)
    stats[3], print_list[3] = _summarize(1, maxDets=20, areaRng='medium')
    stats[4], print_list[4] = _summarize(1, maxDets=20, areaRng='large')
    stats[5], print_list[5] = _summarize(0, maxDets=20)
    stats[6], print_list[6] = _summarize(0, maxDets=20, iouThr=.5)
    stats[7], print_list[7] = _summarize(0, maxDets=20, iouThr=.75)
    stats[8], print_list[8] = _summarize(0, maxDets=20, areaRng='medium')
    stats[9], print_list[9] = _summarize(0, maxDets=20, areaRng='large')

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def save_info(coco_evaluator,
              save_name: str = "record_mAP.txt"):
    # calculate COCO info for all keypoints
    coco_stats, print_coco = summarize(coco_evaluator)

    # 将验证结果保存至txt文件中
    with open(save_name, "w") as f:
        record_lines = ["COCO results:", print_coco]
        f.write("\n".join(record_lines))


# 統計每類關鍵點的平均置信度
def eval_key():
    with open('config.json', 'r') as f:
        config = json.load(f)
    current_time = config["time"]
    file_dir = "./results/val_result/" + current_time + "/"
    os.makedirs(file_dir, exist_ok=True)

    file_name = 'key_results.json'
    with open(file_dir + file_name, 'r') as f:
        results = json.load(f)
    # print((file_dir+file_name))
    # 每類關鍵點的置信度-字典
    key_confidences = {i: [] for i in range(21)}

    # 提取置信度
    for result in results:
        key = result['keypoints']
        score = result['score']
        if score <= 1:
            for i in range(21):
                confidence = key[i*3+2]
                # print(confidence)
                key_confidences[i].append(confidence)

    avg_confidences = {i: np.mean(confidences) if confidences else 0 for i, confidences in key_confidences.items()}

    # 輸出到文件
    with open(file_dir + 'avg_key_confidences.json', 'w') as f:
        json.dump(avg_confidences, f)

    # 绘制平均置信度
    keypoint_names = [
        "L_eye", "R_eye", "nose", "L_ear", "R_ear",
        "L_F_elbow", "R_F_elbow", "L_B_elbow", "R_B_elbow",
        "L_F_knee", "R_F_knee", "L_B_knee", "R_B_knee",
        "L_F_paw", "R_F_paw", "L_B_paw", "R_B_paw",
        "throat", "withers", "tail", "torso"
    ]
    keypoints = list(avg_confidences.keys())
    avg_confidences = list(avg_confidences.values())
    # 柱狀圖
    # plt.bar(keypoints, avg_confidences)
    # 折線圖
    plt.figure(figsize=(10, 6))  # 设置图像大小
    # plt.plot(keypoints, avg_confidences, marker='o')  # 使用折线图并在每个点上显示一个标记
    plt.plot(keypoints, avg_confidences, color='black')
    # 绘制根据置信度改变颜色的点
    for i, conf in enumerate(avg_confidences):
        if conf > 0.6:
            plt.scatter(i, conf, color='green', zorder=5)  # 置信度大于0.6为绿色
        elif 0.5 <= conf <= 0.6:
            plt.scatter(i, conf, color='blue', zorder=5)  # 置信度在0.5到0.6之间为蓝色
        elif 0.4 <= conf < 0.5:
            plt.scatter(i, conf, color='orange', zorder=5)  # 置信度在0.4到0.5之间为黄色
        else:
            plt.scatter(i, conf, color='red', zorder=5)  # 置信度小于0.4为红色
    plt.xlabel('Keypoint Index')
    plt.ylabel('Average Confidence')
    plt.title('Average Keypoint Confidence')
    plt.xticks(keypoints)
    plt.ylim(0, 1)  # 假设置信度在0到1之间
    plt.grid(axis='y')

    # 在右侧附上关键点名称
    init_y = 1.0  # 起始输出位置
    init_x = 21.2
    spacing = 0.05  # 控制文本间距
    for i, name in enumerate(keypoint_names):
        plt.text(init_x, init_y - i * spacing,f"{i}:{name}", fontsize=9, verticalalignment='center')

    # 在每个点上方标注具体值
    for i, conf in enumerate(avg_confidences):
        plt.text(i, conf + 0.02, f"{conf:.2f}", fontsize=9, ha='center')  # +0.02 - 调整值以便在点上方显示

        # 保存图像
    plt.savefig(file_dir + 'avg_key_confidence.png')
    plt.show()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=args.resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # read class_indict
    label_json_path = args.label_json_path
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        person_coco_info = json.load(f)

    data_root = args.data_path

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = CocoKeypoint(data_root, "test", transforms=data_transform["val"], det_json_path=None)
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)

    # create model
    model_type = args.model_type
    if model_type == "HRNET":
        model = HighResolutionNet()
    elif model_type == "CSA":
        model = HighResolutionNet_CSA()
    elif model_type == "CBAM":
        model = HighResolutionNet_CBAM()
    elif model_type == "AHRNET":
        model = HighResolutionNet_Attention()
    else:
        raise ValueError(f"Unsupported attention type: {model_type}")

    # 载入你自己训练好的模型权重
    weights_path = args.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # print(model)
    model.to(device)

    # evaluate on the val dataset
    # current_time = time.strftime("%Y%m%d-%H", time.localtime())
    with open('config.json', 'r') as f:
        config = json.load(f)
    current_time = config["time"]
    results_dir = "./results/val_result/" + current_time + "/"
    # results_file = results_dir + "results.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H"))
    os.makedirs(results_dir, exist_ok=True)

    key_metric = EvalCOCOMetric(val_dataset.coco, "keypoints", results_dir+"key_results.json")
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            images = images.to(device)

            # inference
            outputs = model(images)
            if args.flip:
                flipped_images = transforms.flip_images(images)
                flipped_outputs = model(flipped_images)
                flipped_outputs = transforms.flip_back(flipped_outputs, person_coco_info["flip_pairs"])
                # feature is not aligned, shift flipped heatmap for higher accuracy
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
                outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)

            key_metric.update(targets, outputs)

    key_metric.synchronize_results()
    key_metric.evaluate()

    save_info(key_metric.coco_evaluator, results_dir+"keypoint_record_mAP.txt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--model-type', default='CSA', type=str,
                        help="model type to use: HRNET, CSA, CBAM, or AHRNET")
    # 使用设备类型
    parser.add_argument('--device', default='cuda:0', help='device')

    parser.add_argument('--resize-hw', type=list, default=[256, 192], help="resize for predict")
    # 是否开启图像翻转
    parser.add_argument('--flip', type=bool, default=True, help='whether using flipped train')

    # 数据集的根目录./data/animal
    parser.add_argument('--data-path', default='../Dataset/CAP4000', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights-path', default='./save_weights/20250108-1508/best.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                        help='batch size when validation.')
    # 类别索引和类别名称对应关系
    parser.add_argument('--label-json-path', type=str, default="animal_keypoints.json")
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None
    parser.add_argument('--person-det', type=str, default=None)  # "./COCO_val2017_detections_AP_H_56_person.json"

    args = parser.parse_args()

    main(args)
    eval_key()
