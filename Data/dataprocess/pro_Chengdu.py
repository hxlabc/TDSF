import os
import random

from matplotlib import pyplot as plt

from Celldivide.draw import visiable
from Data.dataprocess.filter_trajectory import tra_filter, list2exel, dis_between_points
from common import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 提取轨迹并生成人口分布
def load_chengdu_tra(txtpath,     # 数据集路径
                     save_path,   # 保存路径
                     p=2e-3,      # 采样概率（人口分布）
                     tralen=50
                     ):
    if not os.path.exists(save_path):
        # 如果不存在，则新建文件夹
        os.makedirs(save_path)

    file_name = txtpath.split("/")[-1].rstrip("train.txt")

    lat_list = []
    lng_list = []

    last_id = 1
    trace_by_id = []
    No = 1
    with open(txtpath, 'r+') as fp:
        for item in fp.readlines():
            item_list = item.split(',')

            id = int(item_list[0])
            lat_i = float(item_list[1])
            lng_i = float(item_list[2])
            timestamp = item_list[4].rstrip('\n')

            ok = cd_lat_min < lat_i < cd_lat_max and cd_lng_min < lng_i < cd_lng_max
            if ok is False:
                continue

            random_number = random.random()
            if random_number <= p:
                lat_list.append(lat_i)
                lng_list.append(lng_i)

            if id != last_id:
                print(id, No)
                if trace_by_id:    # 如果不为空
                    trace_by_id = sorted(trace_by_id, key=lambda x: x[3])  # 轨迹中的位置点按时间戳排序
                    tra_len = 50
                    tra_len = min(tra_len, len(trace_by_id))
                    if len(trace_by_id) >= tralen:
                        trace_by_id = trace_by_id[0:tra_len]  # 抽取前tra_len个位置点
                        save_tra_path = save_path + file_name + "chengdu_{}.txt".format(No)
                        with open(save_tra_path, 'w+') as f:
                            for sublist in trace_by_id:
                                # 将子列表转换为字符串形式
                                sublist_str = ','.join(map(str, sublist))
                                # 写入文件
                                f.write(sublist_str + '\n')
                        f.close()
                        No += 1

                trace_by_id = []
                last_id = id
                if id == 5200:
                    break

            trace_by_id.append([lat_i, lng_i, timestamp, id])

    datalen = len(lat_list)
    return datalen, lat_list, lng_list


# 获取训练相关性矩阵的轨迹集
def get_train(train_path):
    # 取前三天的作为历史轨迹数据集
    history_tra_path = ["/Chengdu/20140822_train.txt",
                        "/Chengdu/20140823_train.txt",
                        "/Chengdu/20140824_train.txt"]
    for hdp in history_tra_path:
        h_tra_path = total_data_path + hdp
        load_chengdu_tra(h_tra_path, train_path)


if __name__ == '__main__':
    txtpath = total_data_path + "/Chengdu/20140824_train.txt"
    save_path = Total_path + "/Data/Chengdu_tra/"
    train_path = output_path + "/Train_Tra/Chengdu_train/"

    # 加载历史轨迹集
    get_train(train_path)

    # 加载当前轨迹集
    # datalen, lat, lng = load_chengdu_tra(txtpath, save_path)

    # list2exel(lat, lng, "Chengdu")
    # visiable(lat, lng, "Chengdu")

    # 过滤轨迹
    # tra_filter(save_path, "Chengdu", minlen=6, name="Chengdu 轨迹长度分布图")
    tra_filter(train_path, "Chengdu", minlen=6)

    # dis_between_points(save_path, True)
    dis_between_points(train_path, True)
