import datetime
import matplotlib.pyplot as plt
import os

def plot_loss_and_lr(train_loss, learning_rate, current_time):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况

        pic_dir = "./results/train_result/" + current_time + "/pic/"
        os.makedirs(pic_dir, exist_ok=True)
        pic_path = os.path.join(pic_dir, "loss_and_lr.png")
        plt.savefig(pic_path)
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP, current_time):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        pic_dir = "./results/train_result/" + current_time + "/pic/"
        os.makedirs(pic_dir, exist_ok=True)
        pic_path = os.path.join(pic_dir, "mAP.png")
        plt.savefig(pic_path)
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)
