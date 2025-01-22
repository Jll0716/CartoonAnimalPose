import os


def my_rename(path):
    file_list = os.listdir(path)
    num_start = 435    # 从文件path目录下的首张开始依次更改
    for i, fi in enumerate(file_list):
        old_dir = os.path.join(path, fi)
        num_start += 1
        filename = "img_"+str(num_start)+".png"   # +str(fi.split(".")[-1])
        new_dir = os.path.join(path, filename)
        try:
            os.rename(old_dir, new_dir)
        except Exception as e:
            print(e)
            print("Failed!")
        else:
            print("Success!")


if __name__ == "__main__":
    path = "/home/jll/Pictures/img_1"
    # path="D:/test/121"
    my_rename(path)
