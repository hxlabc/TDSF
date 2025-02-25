# 给轨迹加噪
import os
import random

import numpy as np
from tqdm import tqdm

from Celldivide.utils import score_p, get_distance, inverseCumulativeGamma, perturb
from common import cor_thred, output_path
from cor_martix import cor_weight, getwaitlist, cor_score, dis_score


# 相关性指数机制
def Cor_Exponential(e,  # 隐私预算
                    slist,  # 三级网格
                    cur_area,  # 当前网格
                    wlist,  # 候选网格集合
                    highcor_area,  # 高相关性网格集合
                    cor_mat  # 相关性矩阵
                    ):
    scores = cor_score(cur_area, wlist, highcor_area, cor_mat)
    # Calculate the probability for each element, based on its score
    probabilities = [score_p(e, 1, score) for score in scores]

    # Normalize the probabilties so they sum to 1
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    # Choose an element from V_area based on the probabilities
    k = np.random.choice(wlist, 1, p=probabilities)[0]  # 选中网格的编号
    noise_area = slist[k]

    # Choose a point from noise_area
    lng_noise = random.uniform(noise_area.pos_up[0], noise_area.pos_down[0])
    lat_noise = random.uniform(noise_area.pos_down[1], noise_area.pos_up[1])

    return lng_noise, lat_noise


# 距离指数机制
def Dis_Exponential(e,  # 隐私预算
                    lat,  # 纬度
                    lng  # 经度
                    ):
    kn = 5
    z = np.random.rand()

    r = inverseCumulativeGamma(e, z) / 111000  # 除以111000是为了将米转化成大概的经纬度
    # 候选点集合
    wlpoints = [[lat + random.uniform(-r, r), lng + random.uniform(-r, r)] for _ in range(kn)]

    scores = dis_score(lat, lng, wlpoints)
    probabilities = [score_p(e, 1, score) for score in scores]
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)
    k = np.random.choice([x for x in range(len(wlpoints))], 1, p=probabilities)[0]
    return wlpoints[k]


# 改进版地理不可区分
def Geo_Indistinguishability(e,  # 隐私预算
                             lat,  # 纬度
                             lng,  # 经度
                             ):
    # # 候选点集合
    # kn = 5
    # wlpoints = []
    # for _ in range(kn):
    #     lng_n, lat_n = perturb(e, lng, lat)
    #     wlpoints.append([lat_n, lng_n])
    #
    # dis_min = 1e5
    # lng_noise = lng
    # lat_noise = lat
    # for g in range(len(wlpoints)):
    #     dis = get_distance(lng, lat, wlpoints[g][1], wlpoints[g][0])
    #     if dis_min > dis:
    #         dis_min = dis
    #         lat_noise, lng_noise = wlpoints[g]

    lng_noise, lat_noise = perturb(e, lng, lat)
    return [lat_noise, lng_noise]


# 给轨迹加噪
def addnoise2tra(Total_tra,  # 所有轨迹信息
                 cor_mat,  # 相关性矩阵
                 epsilon,  # 隐私预算
                 KindsOfArea,  # 各层网格
                 win,  # 滑动窗口大小
                 fnames,  # 轨迹文件名称
                 database,  # 数据集名称
                 method="Geo_I",
                 thred=cor_thred  # 相关性阈值
                 ):
    change_path = "/Noise_Tra/Our/" + database + "/Our_e_{}_win_{}_thred_{}/".format(epsilon, win, thred)
    total_save_path = output_path + change_path
    if not os.path.exists(total_save_path):
        # 如果不存在，则新建文件夹
        os.makedirs(total_save_path)

    alist, blist, slist, indexcell = KindsOfArea

    # 遍历每条轨迹
    for t, trajectory in tqdm(enumerate(Total_tra), desc="轨迹加噪开始"):
        latlng_Noise = []
        Lat = trajectory[0]
        Lng = trajectory[1]
        belongarea = trajectory[2]

        length = len(belongarea)  # 轨迹长度
        Highlist = []  # 用来存储一条轨迹所有点的 highcor_area
        TClist = []  # 用来存储一条轨迹所有点的 highcor_area的相关性总和

        left = 0
        right = min(win, length)
        while left < length:
            neednum = 0  # 该轨迹在某个滑动窗口中需要相关性加噪的点
            # 预遍历每个位置点，计算相应权重
            for i in range(left, right):
                cur_area = belongarea[i]
                highcor_area = []  # 高相关性的区域
                totalcorr = 0  # 高相关性总和
                for j in range(i + 1, length):
                    next_area = belongarea[j]
                    # 如果网格找得到且键值对不为空
                    if cur_area != -1 and cor_mat[cur_area].get(next_area) is not None:
                        cor = cor_mat[cur_area][next_area]
                    else:
                        cor = 0
                    # 相关性低于阈值的过滤掉
                    if cor > thred:
                        highcor_area.append(next_area)
                        totalcorr += cor

                Highlist.append(highcor_area)
                TClist.append(totalcorr)

                # 记录需要相关性加噪的点的数量, 并加噪
                if highcor_area != []:
                    neednum += 1

            # w = neednum / (right - left)   # 计算需要相关性加噪的比例作为权重
            w = 0.6
            e1 = w * epsilon  # 用于相关性加噪
            e2 = (1 - w) * epsilon  # 用于普通加噪

            # 得到相关性加噪的点的权重
            weights = cor_weight(TClist[left:right])

            # 再遍历每个位置点，进行加噪
            for i in range(left, right):
                cur_area = belongarea[i]
                highcor_area = Highlist[i]

                # 进行相关性加噪
                if highcor_area != []:
                    e1i = e1 * weights[i - left]

                    wlist = getwaitlist(cur_area, slist)  # 候选扰动网格

                    if wlist:
                        lng_noise, lat_noise = Cor_Exponential(e1i, slist, cur_area, wlist, highcor_area, cor_mat)
                        latlng_Noise.append([lat_noise, lng_noise])

                    else:
                        if method == "Geo_I":
                            res = Geo_Indistinguishability(e1i, Lat[i], Lng[i])
                        else:
                            res = Dis_Exponential(e1i, Lat[i], Lng[i])
                        latlng_Noise.append(res)

                # 其他点进行普通加噪
                else:
                    # 无需相关性加噪的点进行普通指数机制加噪
                    e2i = e2 / (right - left - neednum)
                    if method == "Geo_I":
                        res = Geo_Indistinguishability(e2i, Lat[i], Lng[i])
                    else:
                        res = Dis_Exponential(e2i, Lat[i], Lng[i])
                    latlng_Noise.append(res)

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


