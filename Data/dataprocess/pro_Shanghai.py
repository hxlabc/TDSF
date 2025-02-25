# 提取轨迹并生成人口分布
import os

from Celldivide.draw import visiable
from Data.dataprocess.filter_trajectory import tra_filter, bernoulli_sampling, list2exel, dis_between_points
from common import *
import random


# 轨迹预处理
def load_shanghai_tra(path,    # 数据集路径
                      save_path,  # 保存路径
                      start=0,
                      intra=3,
                      tralen=-1
                      ):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    numfiles = os.scandir(path)
    # 每一个文件的绝对路径
    for num in numfiles:
        txt_path = path + "/" + num.name
        # print(txt_path)

        latAndlng = []

        with open(txt_path, 'r+') as fp:
            fpr = fp.readlines()

            for item in fpr[start::intra]:
                item_list = item.split(',')
                lat_i = float(item_list[3])
                lng_i = float(item_list[2])

                ok = sh_lat_min < lat_i < sh_lat_max and sh_lng_min < lng_i < sh_lng_max
                if ok is False:
                    continue

                datetime_i = item_list[1]

                latAndlng.append([lat_i, lng_i, datetime_i])

        if latAndlng:
            len1 = len(latAndlng)
            # 将一条长轨迹分成几段，扩充数据集
            tra_len = random.randint(50, 120) if tralen == -1 else tralen
            div_n = len1 // tra_len + 1

            left = 0
            right = min(len1, tra_len)
            for _ in range(div_n):
                save_tra_path = save_path + "{}_{}_".format(start, div_n) + num.name
                is_save = True
                if tralen != -1 and right - left != tralen:
                    is_save = False

                if is_save:
                    with open(save_tra_path, 'w+') as f:
                        for sublist in latAndlng[left:right]:
                            # 将子列表转换为字符串形式
                            sublist_str = ','.join(map(str, sublist))
                            # 写入文件
                            f.write(sublist_str + '\n')
                    f.close()

                left = right
                right = min(len1, right + tra_len)


# 获取训练相关性矩阵的轨迹集
def get_train(path, train_path):
    for start in [0, 1, 2]:
        load_shanghai_tra(path, train_path, start, tralen=50)


# 以下部分用于网格划分，画散点图
def txt2list(path):
    lat = []  # 纬度
    lng = []  # 经度

    numfiles = os.scandir(path)
    # 每一个文件的绝对路径
    for num in numfiles:
        txt_path = path + "/" + num.name
        # print(txt_path)

        with open(txt_path, 'r+') as fp:
            fpr = fp.readlines()
            for item in fpr:
                item_list = item.split(',')
                lat_i = float(item_list[3])
                lng_i = float(item_list[2])

                if sh_lat_min < lat_i < sh_lat_max and sh_lng_min < lng_i < sh_lng_max:
                    lat.append(lat_i)
                    lng.append(lng_i)

    datalen = len(lat)

    return datalen, lat, lng


if __name__ == '__main__':
    p_bs = 0.0065
    path = total_data_path + "/Shanghai"
    save_path = Total_path + "/Data/Shanghai_tra/"
    train_path = output_path + "/Train_Tra/Shanghai_train/"

    # datalen, lat, lng = txt2list(path)
    # datalen, lat, lng = bernoulli_sampling(p_bs, datalen, lat, lng)

    # print(datalen)
    # list2exel(lat, lng, "Shanghai")
    # visiable(lat, lng, "Shanghai")

    # load_shanghai_tra(path, save_path, tralen=50)
    get_train(path, train_path)

    # 过滤轨迹
    # tra_filter(save_path, "Shanghai", minlen=6, name="Shanghai 轨迹长度分布图")
    tra_filter(train_path, "Shanghai", minlen=6)

    # dis_between_points(save_path, True)
    dis_between_points(train_path, True)
