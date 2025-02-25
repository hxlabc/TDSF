import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors

from common import *

plt.rcParams['font.family'] = 'SimHei'


# 数据集可视化
def visiable(lat, lng, Name):

    plt.title("{}抽样数据集".format(Name))
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.scatter(lng, lat, s=2, c='#f0a732', marker='*')

    plt.savefig(Total_path + '/run/RUN_{}/{} 人口分布.jpg'.format(Name, Name), dpi=600)
    plt.show()


# 画分割网格
def draw_areas(alist, name, region, dataBase, lng=None, lat=None):
    fig = plt.figure()

    lat_min, lat_max, lng_min, lng_max = region
    ax = fig.add_subplot(111)
    plt.xlim(lng_min, lng_max)
    plt.ylim(lat_min, lat_max)

    if lng is not None and lat is not None:
        plt.scatter(lng, lat, s=1, c='#f0a732', marker='o')
    for ant in tqdm(alist, desc=name):
        rect = ant.draw_rectangle(edgecolor='#4096c1')
        ax.add_patch(rect)

    savename = Total_path + '/run/RUN_{}/'.format(dataBase) + name + '.jpg'
    plt.savefig(savename, dpi=600)

    return ax


# 画有效网格
def draw_filterareas(slist, name, region, dataBase, lng=None, lat=None):
    fig = plt.figure()

    lat_min, lat_max, lng_min, lng_max = region
    ax = fig.add_subplot(111)
    plt.xlim(lng_min, lng_max)
    plt.ylim(lat_min, lat_max)

    if lng is not None and lat is not None:
        plt.scatter(lng, lat, s=1, c='#f0a732', marker='o')
    for snt in tqdm(slist, desc=name):
        if snt.r == 1:
            rect = snt.draw_rectangle(is_hollow=False, edgecolor='g')
        else:
            rect = snt.draw_rectangle(edgecolor='g')
        ax.add_patch(rect)

    savename = Total_path + '/run/RUN_{}/'.format(dataBase) + name + '.jpg'
    plt.savefig(savename, dpi=600)

    return ax


# 画索引网格示意图
def draw0and1(index, alist, name, region, dataBase):
    fig = plt.figure()

    lat_min, lat_max, lng_min, lng_max = region
    ax = fig.add_subplot(111)
    plt.xlim(lng_min, lng_max)
    plt.ylim(lat_min, lat_max)

    for ant in alist:
        rect = ant.draw_rectangle(edgecolor='g')
        ax.add_patch(rect)

    for ine in index:
        rect = ine.draw_rectangle(edgecolor='b')
        ax.add_patch(rect)

    savename = Total_path + '/run/RUN_{}/'.format(dataBase) + name + '.jpg'
    plt.savefig(savename, dpi=600)

    return ax


# 打印网格面积
def printSarea(slist, name, sarea=None):
    Sarea = []

    for snt in slist:
        Sarea.append(round(np.sqrt(snt.get_square(True)), 0))

    print(name, sorted(Sarea))

    if sarea is not None:
        len1 = len([x for x in Sarea if x < sarea])
        print(len1)
        print(round(len1 / len(slist), 2))


# 画轨迹高相关性点的比例图
def draw_high_point_rate(Total_tra,     # 所有轨迹信息
                         cor_mat,       # 相关性矩阵
                         database,      # 数据集名称
                         thred=cor_thred         # 相关性阈值
                         ):

    rate_list = []
    rate_non0 = []
    # 遍历每条轨迹
    for t, trajectory in tqdm(enumerate(Total_tra), desc="计算需要相关性加噪的比例"):
        belongarea = trajectory[2]

        length = len(belongarea)     # 轨迹长度
        neednum = 0  # 该轨迹相关性加噪的点的数量

        for i in range(length):
            cur_area = belongarea[i]
            for j in range(i+1, length):
                next_area = belongarea[j]
                # 如果网格找得到且键值对不为空
                if cur_area != -1 and cor_mat[cur_area].get(next_area) is not None:
                    cor = cor_mat[cur_area][next_area]
                else:
                    cor = 0
                # 相关性低于阈值的过滤掉
                if cor > thred:
                    # print(cor)
                    neednum += 1
                    break
        rate = neednum / length
        rate_list.append(rate)
        if rate != 0:
            rate_non0.append(rate)

    print("平均占比:", sum(rate_list) / len(rate_list))
    print("去0后的平均占比:", sum(rate_list) / (len(rate_list)-rate_list.count(0)))
    print("总轨迹数:", len(rate_list))
    print("无需相关性加噪轨迹数:", rate_list.count(0))
    # draw_distribution(rate_list, 10, "{}高相关性点占比分布图".format(database), database)
    draw_distribution(rate_non0, 12, "{}高相关性点占比分布图".format(database), database)

# 画轨迹示意图
# def draw_tra(lng_list, lat_list, blist, belongarea, name):
#     fig = plt.figure()
#
#     ax = fig.add_subplot(111)
#     plt.xlim(geo_lng_min, geo_lng_max)
#     plt.ylim(geo_lat_min, geo_lat_max)
#
#     for bnt in blist:
#         rect = bnt.draw_rectangle(edgecolor='g')
#         ax.add_patch(rect)
#
#     for bnt in belongarea:
#         rect = bnt.draw_rectangle(edgecolor='r')
#         ax.add_patch(rect)
#
#     plt.plot(lng_list, lat_list)
#
#     savename = 'D:/123/Pycharm/DP Trajectory Correlation/run/tra_example/' + name
#     plt.savefig(savename, dpi=600)
#
#     return ax


# 画各种分布图
def draw_distribution(clist,       # 集合
                      num_bins,    # 划分区间
                      fname,
                      dataBase
                      ):
    plt.figure()
    counts, bins, patches = plt.hist(clist, bins=num_bins, color='skyblue', edgecolor='white', density=False)

    # 创建渐变色图    # 创建橙红渐变色图
    cmap = mcolors.LinearSegmentedColormap.from_list('gradient', ['lightcoral', 'orange'], N=num_bins)

    # 计算柱子的中心位置
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for i in range(len(counts)):
        # 应用渐变色
        plt.bar(bin_centers[i], counts[i], width=bins[i + 1] - bins[i], color=cmap(i / len(counts)), edgecolor='white')

        # 在每个柱子上方添加标签
        # plt.text(bin_centers[i], counts[i], int(counts[i]), ha='center', va='bottom')

    # 绘制折线图
    plt.plot(bin_centers, counts, marker='^', color='#e63f31', linestyle='-', linewidth=1)
    plt.xlabel('Percentage')
    plt.ylabel('Number of Trajectories')
    # plt.title(fname)

    savename = Total_path + '/run/RUN_{}/{}.jpg'.format(dataBase, fname)
    plt.savefig(savename, dpi=600)
    plt.clf()


# 画实验结果折线图
def draw_metric(x,
                y_list,
                color_list,
                method_list,
                marker_list,
                metric_name,
                database,
                x_label='privacy budget ε'
                ):
    # 绘制折线图
    plt.figure(figsize=(6, 5))  # 设置图的大小，可根据需要调整

    for i in range(len(y_list)-1, -1, -1):
        plt.plot(x, y_list[i], marker=marker_list[i], linestyle='-', color=color_list[i], label=method_list[i])

    # 添加标题和标签
    # plt.title('{}_Comparison'.format(metric_name))
    plt.xlabel(x_label)
    plt.ylabel(metric_name)
    plt.grid(True)  # 添加网格线
    plt.legend()  # 添加图例

    # 显示图形
    plt.tight_layout()  # 自动调整布局

    savename = Total_path + '/run/result/{}/{}_{}.png'.format(metric_name, database, metric_name)
    plt.savefig(savename, dpi=600)
    plt.clf()
