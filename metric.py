"""
    五个指标：
    1、L1距离
    2、L2距离
    3、CP 相关性保护程度：在关键节点上相关性降低的程度
    4、DTW距离
    5、RE 范围查询误差

    计算两个经纬度坐标之间的距离：https://github.com/mapado/haversine/
    参考github: https://github.com/erik-buchholz/RAoPT/blob/main/README.md
"""
import math
import os
import random
from typing import List

import numpy as np
from fastdtw import fastdtw

from tqdm import tqdm

from Celldivide.utils import get_distance
from common import cor_thred
from cor_martix import loaddata_from_xls, get_cormatrix, find_pointarea


# 仅加载轨迹位置信息
def load_trapos(tra_path):
    trafiles = os.scandir(tra_path)
    Total_tra = []
    for tra_item in trafiles:
        txtpath = tra_path + '/' + tra_item.name

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
    return Total_tra


# 加载所属网格信息
def load_tracell(tra_path, alist, blist, slist, indexcell):
    trafiles = os.scandir(tra_path)
    Total_tra = []

    for tra_item in tqdm(trafiles, desc="开始加载轨迹"):
        txtpath = tra_path + '/' + tra_item.name
        belongarea = []
        is_useful = []
        with open(txtpath, 'r+') as fp:
            for item in fp.readlines():
                item_list = item.split(',')
                lat_i = float(item_list[0])
                lng_i = float(item_list[1])

                barea, _ = find_pointarea(lng_i, lat_i, alist, blist, slist, indexcell)
                # 解决查找为空的问题
                if barea is None:
                    belongarea.append(-1)
                    is_useful.append(0)
                else:
                    belongarea.append(barea.No)
                    is_useful.append(barea.r)

        Total_tra.append([[], [], belongarea, is_useful])

    return Total_tra


# 加载网格信息和相关性矩阵
def load_message(tra_path, cell_path_list, atype):
    path0, path1, path2, path3_list = cell_path_list

    # 加载网格信息
    indexcell = loaddata_from_xls(path0, 0)
    alist = loaddata_from_xls(path1, 1)
    blist = loaddata_from_xls(path2, 2)

    lo = 0
    slist = []
    for path3 in path3_list:
        slist += loaddata_from_xls(path3, 3, lo)
        lo = len(slist)

    # 加载轨迹信息
    Total_tra = load_tracell(tra_path, alist, blist, slist, indexcell)

    if atype == "cor":
        # 加载相关性矩阵
        cor_mat = get_cormatrix(Total_tra, slist)
        return cor_mat
    else:
        return indexcell, alist, blist, slist, Total_tra


# 得到L1或者L2分数
def L_distance(oritra_path,  # 原始轨迹路径
               noisetra_path,  # 加噪轨迹路径
               LL  # L1或者L2
               ):
    oritra = load_trapos(oritra_path)
    noisetra = load_trapos(noisetra_path)

    l_score = 0

    # 遍历每一条轨迹
    for i, tra in enumerate(oritra):
        ll = 0  # 每条轨迹的L1分数
        n_tra = noisetra[i]

        lat_list = tra[0]
        lng_list = tra[1]
        nlat_list = n_tra[0]
        nlng_list = n_tra[1]

        length = len(lat_list)

        # 遍历每一个位置点
        for j in range(length):
            if LL == "L1":
                ll += get_distance(lng_list[j], 0, nlng_list[j], 0) + get_distance(0, lat_list[j], 0, nlat_list[j])
            elif LL == "L2":
                ll += get_distance(lng_list[j], lat_list[j], nlng_list[j], nlat_list[j])
        ll /= length
        l_score += ll

    l_score /= len(oritra)
    return l_score


# 计算DTW距离
def DTW_distance(oritra_path,  # 原始轨迹路径
                 noisetra_path,  # 加噪轨迹路径
                 ):
    oritra = load_trapos(oritra_path)
    noisetra = load_trapos(noisetra_path)

    dtw_score = 0

    # 遍历每一条轨迹
    for i, tra in enumerate(oritra):
        n_tra = noisetra[i]

        dtw, _ = fastdtw(tra, n_tra)

        dtw_score += dtw

    dtw_score /= len(oritra)
    return dtw_score


# 相关性保护程度 (每个高相关性点的平均相关性下降程度)
def Cor_protect(noisetra_path,  # 加噪轨迹路径
                indexcell,  # 索引网格
                alist,  # 一级网格
                blist,  # 二级网格
                slist,
                Total_tra,
                cor_mat,
                thred=cor_thred
                ):
    cp = 0
    need_cor = 0  # 需要相关性加噪的轨迹数
    Total_tra_noise = load_tracell(noisetra_path, alist, blist, slist, indexcell)
    # 遍历每条轨迹
    for t, tra in tqdm(enumerate(Total_tra)):
        n_tra = Total_tra_noise[t]

        belongarea_list = tra[2]
        nbelongarea_list = n_tra[2]

        length = len(belongarea_list)  # 轨迹长度

        cor_discount = 0  # 记录单条轨迹的相关性下降程度
        is_cor = False  # 标记是否为需要相关性加噪的轨迹
        high_num = 0  # 轨迹中高相关性点的数量
        # print("**********************************")
        # 遍历每个位置点，计算相关性减小的程度
        for i in range(length):
            cur_area = belongarea_list[i]
            ncur_area = nbelongarea_list[i]
            flag = False  # 标记该点是否为高相关性点

            # printlist = []  # 记录每一个点下降的相关性
            for j in range(i + 1, length):
                next_area = belongarea_list[j]
                cor = 0
                if cur_area != -1 and cor_mat[cur_area].get(next_area) is not None:
                    cor = cor_mat[cur_area][next_area]
                # 相关性低于阈值的过滤掉
                if cor > thred:
                    high_num = high_num + 1 if flag is False else high_num
                    flag = True
                    is_cor = True
                    # 异常处理，防止加噪后所属网格找不到
                    if ncur_area != -1 and cor_mat[ncur_area].get(next_area) is not None:
                        # printlist.append(cor_mat[cur_area][next_area] - cor_mat[ncur_area][next_area])
                        cor_discount += cor_mat[cur_area][next_area] - cor_mat[ncur_area][next_area]
                    else:
                        # printlist.append(cor_mat[cur_area][next_area])
                        cor_discount += cor_mat[cur_area][next_area]

            # print(printlist)

        if high_num != 0:
            cp += cor_discount / high_num
        if is_cor:
            need_cor += 1

    cp /= need_cor
    return cp


# 范围查询误差
class SquareQuery:
    def __init__(self,
                 min_lng: float,
                 min_lat: float,
                 max_lng: float,
                 max_lat: float,
                 edge: float):
        # Randomly select center
        center_lng = random.random() * (max_lng - min_lng) + min_lng
        center_lat = random.random() * (max_lat - min_lat) + min_lat
        self.center = (center_lng, center_lat)

        self.left_lng = center_lng - edge / 2
        self.up_lat = center_lat + edge / 2
        self.right_lng = center_lng + edge / 2
        self.down_lat = center_lat - edge / 2

    def in_square(self, point):
        return self.left_lng <= point[0] <= self.right_lng and self.down_lat <= point[1] <= self.up_lat

    def point_query(self, db):
        count = 0
        for tra in db:
            lat_list = tra[0]
            lng_list = tra[1]
            for j in range(len(lat_list)):
                point = [lng_list[j], lat_list[j]]
                if self.in_square(point):
                    count += 1

        return count


# 生成询问
def genrateQueries(min_lng, min_lat, max_lng, max_lat, edge, nums, seed):
    random.seed(seed)
    QList = []
    edge /= 111000  # 将米大致转为经纬度长度

    ad = 8e-3
    # 调整范围，避免越界
    min_lng += ad
    min_lat += ad
    max_lng -= ad
    max_lat -= ad

    for i in range(nums):
        QList.append(SquareQuery(min_lng, min_lat, max_lng, max_lat, edge))
    return QList


# 计算RE
def calculate_RE(oritra_path,  # 原始轨迹路径
                 noisetra_path,  # 加噪轨迹路径
                 queries: List[SquareQuery],
                 sanity_bound=0.01):
    oritra = load_trapos(oritra_path)
    noisetra = load_trapos(noisetra_path)

    actual_ans = list()
    syn_ans = list()

    D = len(oritra)

    for q in tqdm(queries, desc="正在计算RE"):
        actual_ans.append(q.point_query(oritra))
        syn_ans.append(q.point_query(noisetra))

    actual_ans = np.asarray(actual_ans)
    syn_ans = np.asarray(syn_ans)

    # Error = |actual-syn| / max{actual, 1% * len(db)}
    numerator = np.abs(actual_ans - syn_ans)
    # numerator = syn_ans - actual_ans
    denominator = np.asarray([max(actual_ans[i], D * sanity_bound) for i in range(len(actual_ans))])
    # denominator = actual_ans

    error = numerator / denominator

    return np.mean(error)


# 查询误差（废弃指标）
def QE(oritra_path,  # 原始轨迹路径
       noisetra_path,  # 加噪轨迹路径
       maxlen,  # 子轨迹最长长度
       mindis  # 允许的最大偏移距离（单位：米）
       ):
    oritra = load_trapos(oritra_path)
    noisetra = load_trapos(noisetra_path)

    count = 0  # 子轨迹成功查询数
    # 遍历每一条轨迹
    for i, tra in tqdm(enumerate(oritra)):
        n_tra = noisetra[i]

        lat_list = tra[0]
        lng_list = tra[1]
        nlat_list = n_tra[0]
        nlng_list = n_tra[1]

        length = len(lat_list)
        sublen = min(length, maxlen)  # 子轨迹长度
        l = random.randint(0, length - sublen)
        r = l + sublen

        flag = True
        # 遍历子轨迹
        for j in range(l, r):
            dis = get_distance(lng_list[j], lat_list[j], nlng_list[j], nlat_list[j])
            if dis > mindis:
                flag = False
                break

        if flag:
            count += 1

    qe = (len(oritra) - count) / max(len(oritra), 0.001)

    return qe
