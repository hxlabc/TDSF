# 数据集的 维度 经度 高度 日期 时间
#
#
import os

import matplotlib.pyplot as plt

from Celldivide.draw import visiable
from Data.dataprocess.filter_trajectory import bernoulli_sampling, list2exel, dis_between_points, tra_filter
from common import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 根据轨迹长度获取采样间隔
def get_intra(sdd):
    if sdd <= 200:
        intra = 2
    elif sdd <= 500:
        intra = 5
    elif sdd <= 1000:
        intra = 10
    elif sdd <= 3000:
        intra = 15
    elif sdd <= 5000:
        intra = 25
    else:
        intra = -1
    return intra


# 解析plt文件，提取用户轨迹
def plt2txt(path, save_path, start=6, tralen=-1):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    numfiles = os.scandir(path)
    # 每一个文件的绝对路径
    for num in numfiles:
        path_item = path + "/" + num.name + "/Trajectory"
        print(path_item)
        pltsfiles = os.scandir(path_item)

        for pltitem in pltsfiles:
            plt_path = path_item + '/' + pltitem.name

            latAndlng = []
            is_save = True

            with open(plt_path, 'r+') as fp:
                fpr = fp.readlines()
                sdd = len(fpr) - 6
                intra = get_intra(sdd)

                if intra == -1:
                    continue

                for item in fpr[start::intra]:
                    item_list = item.split(',')
                    lat_i = float(item_list[0])
                    lng_i = float(item_list[1])

                    ok = geo_lat_min < lat_i < geo_lat_max and geo_lng_min < lng_i < geo_lng_max
                    if ok is False:
                        is_save = False
                        break

                    date_i = item_list[5]
                    time_i = item_list[6].rstrip('\n')

                    latAndlng.append([lat_i, lng_i, date_i, time_i])

            # 是否限制长度: -1不限长度， 具体值则限长度
            if tralen != -1:
                if len(latAndlng) >= tralen and is_save is True:
                    latAndlng = latAndlng[0:tralen]
                else:
                    is_save = False

            if latAndlng and is_save:
                save_tra_path = save_path + num.name + "_" + str(start) + pltitem.name.replace('.plt', '.txt')
                with open(save_tra_path, 'w+') as f:
                    for sublist in latAndlng:
                        # 将子列表转换为字符串形式
                        sublist_str = ','.join(map(str, sublist))
                        # 写入文件
                        f.write(sublist_str + '\n')
                f.close()


# 获取训练相关性矩阵的轨迹集
def get_train(path, train_path):
    for start in [6, 8, 10]:
        plt2txt(path, train_path, start, tralen=50)


# 以下部分用于网格划分，画散点图
def plt2list(path):
    lat = []  # 纬度
    lng = []  # 经度

    numfiles = os.scandir(path)

    # 每一个文件的绝对路径
    for num in numfiles:
        path_item = path + "/" + num.name + "/Trajectory"
        print(path_item)
        pltsfiles = os.scandir(path_item)
        for pltitem in pltsfiles:
            plt_path = path_item + '/' + pltitem.name
            with open(plt_path, 'r+') as fp:
                for item in fp.readlines()[6::50]:
                    item_list = item.split(',')
                    lat_i = float(item_list[0])
                    lng_i = float(item_list[1])

                    # 选取位于北京的位置点
                    if geo_lat_min < lat_i < geo_lat_max and geo_lng_min < lng_i < geo_lng_max:
                        lat.append(lat_i)
                        lng.append(lng_i)

    datalen = len(lat)

    return datalen, lat, lng


if __name__ == '__main__':
    # 采样概率
    p_bs = 0.06
    path = total_data_path + "/Geolife Trajectories 1.3" + "/Data"
    save_path = Total_path + "/Data/Geolife_tra/"
    train_path = output_path + "/Train_Tra/Geolife_train/"

    # 生成轨迹，保存到txt文件
    # plt2txt(path, save_path, tralen=50)

    # 训练轨迹，保存到txt文件
    # get_train(path, train_path)

    # 生产数据点集
    datalen, lat, lng = plt2list(path)
    print(datalen)

    # datalen, lat, lng = bernoulli_sampling(p_bs, datalen, lat, lng)
    # list2exel(lat, lng, "Geolife")
    # visiable(lat, lng, "Geolife")

    # 过滤轨迹
    # tra_filter(save_path, "Geolife", minlen=6, name="Geolife 轨迹长度分布图")
    # tra_filter(train_path, "Geolife", minlen=6, maxlen=140)

    # dis_between_points(save_path, True)
    # dis_between_points(train_path, True)
    # print(avg_dis)




