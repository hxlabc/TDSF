import math
import os

import pandas as pd
from tqdm import tqdm

from Celldivide.My_worker import Area
from Celldivide.utils import get_distance, tran2corr


# 加载网格信息
def loaddata_from_xls(fname, level, start_i=0):
    df = pd.read_excel(fname)
    x_up = df['x_up'].tolist()
    y_up = df['y_up'].tolist()
    x_down = df['x_down'].tolist()
    y_down = df['y_down'].tolist()
    if level == 1 or level == 0:
        include_parts = df['所包含网格编号'].tolist()
    elif level == 2:
        neibor = df["邻居"].tolist()
        include_parts = df['所包含网格编号'].tolist()
    else:  # level == 3
        neibor = df["邻居"].tolist()
        rlist = df["有效标记"].tolist()

    Areas = []

    for i in range(len(x_up)):
        area = Area([x_up[i], y_up[i]], [x_down[i], y_down[i]], level)
        area.No = start_i + i
        if level == 1 or level == 0:
            area.include_parts = [int(val) for val in include_parts[i].split(",")]
        elif level == 2:
            area.neighbors = [int(val) for val in neibor[i].split(",")]
            area.include_parts = [int(val) for val in include_parts[i].split(",")]
        else:  # level == 3
            area.neighbors = [int(val) for val in neibor[i].split(",")]
            area.r = rlist[i]
        Areas.append(area)
    return Areas


# 加载轨迹数据集
def load_tradata(tra_path, alist, blist, slist, indexcell):
    trafiles = os.scandir(tra_path)
    Total_tra = []
    fnames = []

    for tra_item in tqdm(trafiles, desc="开始加载轨迹"):
        ok = 1
        txtpath = tra_path + '/' + tra_item.name
        fnames.append(tra_item.name)
        # print(txtpath)

        # 存轨迹的横纵坐标
        lat_list = []
        lng_list = []
        belongarea = []
        is_useful = []
        with open(txtpath, 'r+') as fp:
            for item in fp.readlines():
                item_list = item.split(',')
                lat_i = float(item_list[0])
                lng_i = float(item_list[1])

                barea, _ = find_pointarea(lng_i, lat_i, alist, blist, slist, indexcell)

                lat_list.append(lat_i)
                lng_list.append(lng_i)
                # 解决查找为空的问题
                if barea is not None:
                    belongarea.append(barea.No)
                    is_useful.append(barea.r)
                else:
                    belongarea.append(-1)
                    is_useful.append(0)

        if ok:
            Total_tra.append([lat_list, lng_list, belongarea, is_useful])
        # draw_tra(lng_list, lat_list, blist, belongarea, tra_item.name.replace('.txt', '.jpg'))

    return Total_tra, fnames   # [[纬度，经度，所属三级网格, 是否有效]...], 文件名称


# 找到该位置点所属的三级网格
def find_pointarea(pos_x, pos_y, A_area, B_area, C_area, indexcell):
    # 先找索引网格
    for incel in indexcell:
        if incel.is_inArea(pos_x, pos_y):
            # 找到后找索引网格里的一级网格
            for k in incel.include_parts:
                if A_area[k].is_inArea(pos_x, pos_y):
                    # 找到后找一级网格里的二级网格
                    for i in A_area[k].include_parts:
                        if B_area[i].is_inArea(pos_x, pos_y):
                            # 找到后找一级网格里的二级网格
                            for j in B_area[i].include_parts:
                                if C_area[j].is_inArea(pos_x, pos_y, True):
                                    return C_area[j], j
    return None, -1


# 计算相关性矩阵
def get_cormatrix(Train_tra, slist):
    # 相关性矩阵
    n = len(slist)
    cor_mat = [{} for _ in range(n)]

    for trajectory in tqdm(Train_tra, desc="开始计算转移相关性矩阵"):
        belongarea = trajectory[2]  # 每个坐标点所属三级网格的标号
        is_useful = trajectory[3]   # 是否有效
        length = len(belongarea)
        for i in range(length):
            cur_area = belongarea[i]
            # 有效网格才进行转移概率计算
            if is_useful[i] == 1:
                for j in range(i+1, length):
                    next_area = belongarea[j]
                    if is_useful[j] == 1:
                        if next_area not in cor_mat[cur_area]:
                            cor_mat[cur_area][next_area] = 1
                        else:
                            cor_mat[cur_area][next_area] += 1

    # 归一化
    for k in range(len(cor_mat)):
        if cor_mat[k] != {}:
            summ = sum(cor_mat[k].values())
            maxx = max(cor_mat[k].values())

            for key in cor_mat[k].keys():
                tranpro = cor_mat[k][key] / summ
                cor_mat[k][key] = tran2corr(tranpro, maxx)

    return cor_mat


# 得到候选扰动网格
def getwaitlist(cur_area, blist):
    neibors = blist[cur_area].neighbors

    wlist = []
    for nei in neibors:
        # 无效网格不作为候选集
        if blist[nei].r == 1:
            wlist.append(nei)

    return wlist


# 指数机制的得分函数 —— 相关性
def cor_score(cur_area,     # 当前需要扰动的网格
              wlist,        # 候选扰动网格
              highcor_area,    # 与该网格存在高相关性的网格
              cor_mat          # 相关性矩阵
              ):
    # 记录每个候选网格的分数值
    scores = []

    for wl_area in wlist:
        score = 0
        # 记录 将cur_area扰动成wl_area后，与每个h_area相关性变化的差值 之和
        for h_area in highcor_area:
            if cor_mat[wl_area].get(h_area) is not None:
                score += cor_mat[cur_area][h_area] - cor_mat[wl_area][h_area]
            else:
                score += cor_mat[cur_area][h_area]

        scores.append(score)

    return scores    # 该评分函数灵敏度为1


# 指数机制的得分函数 —— 距离
def dis_score(lat,
              lng,
              wlpoints):
    # 记录每个候选点的分数值
    scores = []
    disum = 0

    for pos in wlpoints:
        ss = get_distance(lng, lat, pos[1], pos[0])
        scores.append(ss)
        disum += ss

    for i in range(len(scores)):
        # scores[i] = 1 - scores[i] / disum
        # 更换评分函数，拉开各个评分的差距
        scores[i] = math.exp(-10 * (scores[i] / disum))

    return scores    # 该评分函数敏感度为1


# 相关性加噪的点的权重计算 （动态分配隐私预算）
def cor_weight(TClist):
    """
        相关性总分越大，权重越大，隐私预算越大
        因为在指数机制中隐私预算越大会增大你所希望选到元素的概率
    """

    summ = sum(TClist)
    if summ != 0:
        weights = [tc/summ for tc in TClist]
    else:
        weights = []

    return weights


# 过滤掉处于不可达网格的点
# def select_achievable_points(wlpoints, alist, blist, slist, indexcell):
#     new_wlpoints = []
#     for pos in wlpoints:
#         area, _ = find_pointarea(pos[1], pos[0], alist, blist, slist, indexcell)
#         if area is not None and area.r == 1:
#             new_wlpoints.append(pos)
#
#     if not new_wlpoints:
#         return wlpoints, False
#     else:
#         return new_wlpoints, True



