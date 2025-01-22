import json
import os
import datetime
import time

import torch
from torch.utils import data
import numpy as np

import math
import transforms
from model import HighResolutionNet
from model.ahrnet import HighResolutionNet_Attention
from model.csa_net import HighResolutionNet_CSA
from model.cbam import HighResolutionNet_CBAM
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils


def calculate_lambda(t, T, growth_rate):
    return 0.95 + 0.05 / (1 + math.exp(-growth_rate * (t - T/2)))


def create_model(num_joints, load_pretrain_weights=True):
    model_type = args.model_type
    if model_type == "HRNET":
        model = HighResolutionNet(base_channel=32, num_joints=num_joints)
    elif model_type == "CSA":
        model = HighResolutionNet_CSA(base_channel=32, num_joints=num_joints)
    elif model_type == "CBAM":
        model = HighResolutionNet_CBAM(base_channel=32, num_joints=num_joints)
    elif model_type == "AHRNET":
        model = HighResolutionNet_Attention(base_channel=32, num_joints=num_joints)
    else:
        raise ValueError(f"Unsupported attention type: {model_type}")
    if load_pretrain_weights:
        # 载入预训练模型权重
        # 链接:https://pan.baidu.com/s/1Lu6mMAWfm_8GGykttFMpVw 提取码:f43o
        # weights_dict = torch.load("./hrnet_w32.pth", map_location='cpu')
        # weights_dict = torch.load("./checkpoints/rtmpose-s_humanart-256x192.pth", map_location='cpu')
        weights_dict = torch.load("../Dataset/weight_dict/cspnext-s_imagenet_600e-ea671761.pth", torch.device('cuda:0'))
        # HumanArt
        # weights_dict = torch.load("../Dataset/weight_dict/animalpose-apk10-mAP566-model299-20240911.pth",torch.device('cuda:0'))  # noqa
        # APK-10

        for k in list(weights_dict.keys()):
            # 如果载入的是imagenet权重，就删除无用权重
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]

            # 如果载入的是coco权重，对比下num_joints，如果不相等就删除
            if "final_layer" in k:
                if weights_dict[k].shape[0] != num_joints:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys)

    return model


def main(args):
    s_time = time.time()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    current_time = time.strftime("%Y%m%d-%H%M", time.localtime())
    config = {
        "time": current_time,
        # 其他训练参数可以添加在这里
    }
    with open('config.json', 'w') as f:
        json.dump(config, f)
        # 记录下此时的文件名

    # 加载config文件
    with open('config.json', 'r') as f:
        config = json.load(f)
    current_time = config["time"]

    results_dir = "./results/train_result/" + current_time + "/coco_info/"
    # source_dataset_name = os.path.basename(args.source_dataset.rstrip('/'))
    # target_dataset_name = os.path.basename(args.target_dataset.rstrip('/'))
    # results_dir = f"./results/train_result/{args.model_type}-{target_dataset_name}-{source_dataset_name}-{args.epochs}-{current_time}/coco_info/"
    results_file = results_dir + "results.txt"
    os.makedirs(results_dir, exist_ok=True)

    with open(args.keypoints_path, "r") as f:
        person_kps_info = json.load(f)

    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))
    data_transform = {
        "train": transforms.Compose([
            transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # data_root = args.data_path
    target_data = args.target_dataset
    source_data = args.source_dataset

    # load train data set
    # coco2017 -> annotations -> person_keypoints_train2017.json
    source_train_dataset = CocoKeypoint(source_data, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)
    target_train_dataset = CocoKeypoint(target_data,"train",transforms=data_transform["train"],fixed_size=args.fixed_size)
    source_train_num = len(source_train_dataset)
    target_train_num = len(target_train_dataset)
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    source_train_data_loader = data.DataLoader(source_train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=source_train_dataset.collate_fn)
    target_train_data_loader = data.DataLoader(target_train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=target_train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> person_keypoints_val2017.json
    source_val_dataset = CocoKeypoint(source_data, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)
    target_val_dataset = CocoKeypoint(target_data, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                                      det_json_path=args.person_det)
    source_val_num = len(source_val_dataset)
    target_val_num = len(target_val_dataset)
    source_val_data_loader = data.DataLoader(source_val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=source_val_dataset.collate_fn)
    target_val_data_loader = data.DataLoader(target_val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=target_val_dataset.collate_fn)
    # print("using {} images for training, {} images for validation.".format(train_num, val_num))
    # create model
    model = create_model(num_joints=args.num_joints)
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    source_loss = []
    target_loss = []

    # 保存模型权重的地址，在本次训练中地址唯一。
    # current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    folder_name = "./save_weights/" + current_time
    os.makedirs(folder_name, exist_ok=True)

    # 中点和增长速率参数
    T = 100  # T/2
    growth_rate = 0.3  # k
    combined_loss = None
    best_metric = -float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss_source, lr_source = utils.train_one_epoch_joint_loss(model, optimizer, source_train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        source_loss.append(mean_loss_source.item())


        # target_loss_weight = 0.6
        target_loss_weight = calculate_lambda(epoch, T, growth_rate)
        # 输出当前 λ 值，方便跟踪
        # print(f"Epoch {epoch}, Target Loss Weight (λ): {target_loss_weight:.4f}")

        mean_loss_target, lr_target = utils.train_one_epoch_joint_loss(model, optimizer, target_train_data_loader,
                                                            device=device, epoch=epoch,lamuda=target_loss_weight,
                                                            print_freq=50, warmup=True,
                                                            scaler=scaler)
        target_loss.append(mean_loss_target.item())
        combined_loss = mean_loss_source + target_loss_weight * mean_loss_target

        # combined_loss = torch.tensor(combined_loss, requires_grad=True, device=device)
        # 保存损失值，更新 train_loss
        train_loss.append(combined_loss)
        # train_loss[-1] += target_loss_weight * mean_loss_target.item()
        # train_loss[-1] += mean_loss_source.item()
        # 可选：记录源域和目标域的独立损失，便于分析
        # print(f"Epoch {epoch}, Source Loss: {mean_loss_source.item():.4f}, "
        #       f"Target Loss: {mean_loss_target.item():.4f}, Combined Loss: {combined_loss:.4f}")

        learning_rate.append(lr_source)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, target_val_data_loader, device=device,
                                   flip=True, flip_pairs=person_kps_info["flip_pairs"])

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [combined_loss.item()]] + [f"{lr_target:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # @0.5 mAP

        # save weights
        val_ap = coco_info[1]
        if val_ap > best_metric:
            best_metric = val_ap
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if args.amp:
                save_files['scaler_state_dict'] = scaler.state_dict()
            file_path = os.path.join(folder_name, "best.pth".format(epoch))
            torch.save(save_files, file_path)

        # save weights
        if (epoch + 1) == args.epochs:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            file_path = os.path.join(folder_name, "model-{}.pth".format(epoch))
            torch.save(save_files, file_path)

    e_time = time.time()
    t_time = e_time - s_time
    t_time_str = str(datetime.timedelta(seconds=int(t_time)))
    print(f"total training time:{t_time_str}")

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate, current_time)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map, current_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--model-type',default='CSA',type=str,help="model type to use: HRNET, CSA, CBAM, or AHRNET")
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(coco2017)
    # parser.add_argument('--data-path', default='/data/coco2017', help='dataset')
    # parser.add_argument('--data-path', default='../Dataset/AnimalPose_HR', help='dataset')
    parser.add_argument('--source-dataset', default='../Dataset/AnimalPose_HR', help='dataset')
    parser.add_argument('--target-dataset', default='../Dataset/CAP4000', help='dataset')
    # COCO数据集人体关键点信息
    # parser.add_argument('--keypoints-path', default="./person_keypoints.json", type=str,
    #                         help='person_keypoints.json path')
    parser.add_argument('--keypoints-path', default="./animal_keypoints.json", type=str,
                        help='animal_keypoints.json path')

    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[256, 192], nargs='+', type=int, help='input size')
    # keypoints点数
    # parser.add_argument('--num-joints', default=17, type=int, help='num_joints')
    parser.add_argument('--num-joints', default=21, type=int, help='num_joints')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')

    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数,210
    parser.add_argument('--epochs', default=210, type=int, metavar='N',
                        help='number of total epochs to run')

    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[170, 200], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument('--lr', default=0.0015, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch-size', default=48, type=int, metavar='N',
                        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
