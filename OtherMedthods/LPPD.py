# 加载轨迹数据集
import math
import os
import random

import networkx as nx
import numpy as np
from tqdm import tqdm

from Celldivide.utils import get_distance, addVectorToPos
from common import *


# 加载轨迹数据集
def load_tradata_lppd(tra_path):
    trafiles = os.scandir(tra_path)
    Total_tra = []
    fnames = []

    for tra_item in tqdm(trafiles, desc="开始加载轨迹"):
        txtpath = tra_path + '/' + tra_item.name
        fnames.append(tra_item.name)
        # print(txtpath)

        # 存轨迹的横纵坐标
        lat_list = []
        lng_list = []
        with open(txtpath, 'r+') as fp:
            for item in fp.readlines():
                item_list = item.split(',')
                lat_i = float(item_list[0])
                lng_i = float(item_list[1])
                lat_list.append(lat_i)
                lng_list.append(lng_i)

        Total_tra.append([lat_list, lng_list])

    return Total_tra, fnames


# 计算基于距离的相关性
def cor_base_dis(pos1, pos2):
    lng1, lat1 = pos1
    lng2, lat2 = pos2

    dis_lng = get_distance(lng1, 0, lng2, 0) / 100
    dis_lat = get_distance(0, lat1, 0, lat2) / 100
    cor_lng = np.exp(-dis_lng)
    cor_lat = np.exp(-dis_lat)

    return cor_lng * cor_lat


# 计算相关性矩阵，即边的权值
def calculate_edges(lng_list, lat_list, thred):
    track = np.column_stack([lng_list, lat_list])
    length = len(track)

    martix = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(length):
        for j in range(i+1, length):
            cbd = cor_base_dis(track[i], track[j])
            # 过滤掉低相关性的边
            if cbd > thred:
                martix[i][j] = cbd
                martix[j][i] = cbd

    return martix


# 相关性聚类
def Cor_clustering(tra_length, martix):
    rest = [1 for _ in range(tra_length)]   # 轨迹中剩下的点

    clusterClass = []   # 存聚类结果, 位置点的下标   type: [list, list, list, ...]

    while rest.count(0) != tra_length:
        # 创建一个带权重的无向图
        G = nx.Graph()
        edges = []
        for i in range(tra_length):
            for j in range(i+1, tra_length):
                # 如果点和边都存在
                if rest[i] == 1 and rest[j] == 1 and martix[i][j] != 0:
                    edges.append((i, j, {'weight': martix[i][j]}))

        # 如果已经不存在边了则退出循环
        if not edges:
            break

        G.add_edges_from(edges)

        # 找到最大权重的完全子图
        cliques = list(nx.find_cliques(G))
        max_weight_clique = max(cliques, key=lambda c: sum(G[u][v]['weight'] for u, v in nx.utils.pairwise(c)))
        clusterClass.append(max_weight_clique)

        # 删掉已经选中的点和边
        for z in max_weight_clique:
            rest[z] = 0     # 只需删除点，上面的判断已包含删除边

    return clusterClass


# 计算每个类别的权重
def get_cluster_weight(clusterClass, martix):
    cs_list = []
    for cluster in clusterClass:
        length = len(cluster)
        cs = 0
        # 计算每个类中边的权值总和
        for i in range(length):
            for j in range(i+1, length):
                cs += martix[cluster[i]][cluster[j]]
        cs_list.append(cs)

    csum = sum(cs_list)
    weight = [cs / csum for cs in cs_list]

    return weight


# 计算距离权重
def get_dis_weight(trajectory):
    dis_list = []
    for i in range(1, len(trajectory)):
        last_lng = trajectory[i-1][1]
        last_lat = trajectory[i-1][0]
        cur_lng = trajectory[i-1][1]
        cur_lat = trajectory[i-1][0]

        dis = get_distance(last_lng, last_lat, cur_lng, cur_lat)
        dis_list.append(max(dis, 5))   # 防止权重为0

    if dis_list:
        dis_list.append(dis_list[-1])  # 再补充一个，论文里有点奇怪，改进一下

    dsum = sum(dis_list)
    if dsum != 0:
        weight = [ds / dsum for ds in dis_list]
    else:
        weight = [1]   # 如果只有一个点权重直接设为1

    return weight


def get_r_pi(lng_dis, lat_dis, U):
    lng_dis *= U[0]
    lat_dis *= U[1]
    r = np.sqrt(lng_dis**2+lat_dis**2)
    if lng_dis == 0:
        return r, math.pi/2
    else:
        tan_x = lat_dis / lng_dis
        radians = math.atan(tan_x)
        return r, radians


# 给轨迹加噪
def addnoise_lppd(Total_tra,  # 所有轨迹信息
                  δ,  # 隐私预算 δ
                  fnames,  # 轨迹文件名称
                  win,  # 滑动窗口的大小
                  β1,
                  β2,
                  thred,
                  database,     # 数据集名称
                  pb=0.9        # 柏努利分布概率
                  ):
    change_path = "/Noise_Tra/LPPD/" + database + "/LPPD_e_{}_win_{}/".format(δ, win)
    total_save_path = output_path + change_path
    if not os.path.exists(total_save_path):
        # 如果不存在，则新建文件夹
        os.makedirs(total_save_path)

    # 遍历每条轨迹
    for t, trajectory in tqdm(enumerate(Total_tra), desc="轨迹加噪开始"):

        Lat = trajectory[0]
        Lng = trajectory[1]
        length = len(Lat)
        latlng_Noise = np.column_stack([Lat, Lng])   # 用原始轨迹初始化噪声轨迹

        e_cor = δ * β1     # 相关性加噪的权重
        e_dis = δ * β2     # 距离分数加噪的权重

        e_cor = e_cor * (length / win)     # 为了满足滑动窗口特质增大隐私预算
        martix = calculate_edges(Lng, Lat, thred)              # 获得相关性矩阵
        clusterClass = Cor_clustering(len(Lng), martix)     # 聚成高相关性图类
        weights = get_cluster_weight(clusterClass, martix)      # 获得每个类的权重

        for c, cluster in enumerate(clusterClass):
            e_c = e_cor * weights[c] / len(cluster)   # 计算每个点的隐私预算
            # 每个类里面进行相关性加噪
            for cl in cluster:
                lng_dis = np.random.laplace(loc=0, scale=1 / e_c)   # 灵敏度为 1
                lat_dis = np.random.laplace(loc=0, scale=1 / e_c)
                U = [np.random.binomial(n=1, p=pb), np.random.binomial(n=1, p=pb)]

                r, radians = get_r_pi(lng_dis, lat_dis, U)
                latlng_Noise[cl][1], latlng_Noise[cl][0] = addVectorToPos(Lng[cl], Lat[cl], r, radians)


        # 进行距离分数的加噪
        left = 0
        right = min(win, length)
        while left < length:
            for i in range(left, right):
                trace_win = latlng_Noise[left:right]   # 获取一段轨迹窗口
                dis_weights = get_dis_weight(trace_win)
                e_d = e_dis * dis_weights[i-left]

                lng_dis = np.random.laplace(loc=0, scale=2 / e_d)  # 灵敏度为 2
                lat_dis = np.random.laplace(loc=0, scale=2 / e_d)
                U = [np.random.binomial(n=1, p=pb), np.random.binomial(n=1, p=pb)]

                r, radians = get_r_pi(lng_dis, lat_dis, U)
                latlng_Noise[i][1], latlng_Noise[i][0] = addVectorToPos(Lng[i], Lat[i], r, radians)

            # 更新窗口滑动
            left = right
            right = min(right + win, length)

        # 将加完噪的轨迹存入txt文件
        save_tra_path = total_save_path + fnames[t]
        # print(fnames[t])
        with open(save_tra_path, 'w+') as f:
            for sublist in latlng_Noise:
                # 将子列表转换为字符串形式
                sublist_str = ','.join(map(str, sublist))
                # 写入文件
                f.write(sublist_str + '\n')
        f.close()

