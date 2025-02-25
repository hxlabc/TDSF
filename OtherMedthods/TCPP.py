import math
import os
import random

from scipy.spatial import Voronoi
from sklearn.cluster import KMeans
from tqdm import tqdm

from Celldivide.utils import get_distance, sim, addVectorToPos
from Kalman import kf_trajectory
from common import *
from cor_martix import find_pointarea


# 构建一条轨迹所对应的Voronoi网格
def getvoronoi(lng_list,   #轨迹的经度集合
               lat_list,   #轨迹的纬度集合
               slng,       #轨迹中经度min
               elng,       #轨迹中经度max
               slat,       #轨迹中纬度min
               elat,       #轨迹中纬度max
               n=6         # 敏感区域个数（Voronoi网格个数）
               ):
    # 生成随机点集
    points = []
    """
    修改: 根据每条轨迹位置点的聚集程度来选分割点
    """
    if slng == elng:
        slng -= 0.0001
        elng += 0.0001
    if slat == elat:
        slat -= 0.0001
        elat += 0.0001

    for _ in range(n):
        lng = random.uniform(slng, elng)
        lat = random.uniform(slat, elat)
        points.append([lng, lat])

    # 创建 Voronoi 图
    vor = Voronoi(points)
    belongVoronoi = []    # 每个轨迹点所属Voronoi网格的索引

    for i in range(len(lng_list)):
        point = np.array([lng_list[i], lat_list[i]])
        # 获取包含指定点的 Voronoi 区域的索引
        region_index = np.argmin(np.linalg.norm(vor.points - point, axis=1))
        belongVoronoi.append(region_index)

    return vor.points, belongVoronoi


# 给轨迹聚类并添加索引，提升算法效率
def Kmean_index(Total_tra, nums=20):
    tra_list = []
    for trajectory in Total_tra:
        Lat = trajectory[0]
        Lng = trajectory[1]
        tra_list.append([np.mean(Lat), np.mean(Lng)])

    num_clusters = len(tra_list) // nums
    # print("聚类数：", num_clusters)

    # KMeans聚类
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(tra_list)

    # 输出聚类结果
    labels = kmeans.labels_.tolist()

    return num_clusters, labels


# 初步过滤轨迹
def filterByEuclidean(tr,        # 目标轨迹
                      Total_tra,    # 轨迹数据集
                      thred,        # 距离阈值
                      kind,         # trace 所属的种类
                      labels        # Total_tra中每条轨迹所属的种类
                      ):
    Da = [tr]
    Lat2 = Total_tra[tr][0]
    Lng2 = Total_tra[tr][1]
    tralen = len(Lat2)
    for t, trajectory in enumerate(Total_tra):
        if labels[t] == kind and t != tr:
            Lat1 = trajectory[0]
            Lng1 = trajectory[1]

            edis = 0
            for i in range(tralen):
                edis += get_distance(Lng1[i], Lat1[i], Lng2[i], Lat2[i])
            edis /= tralen

            if edis < thred:
                Da.append(t)

            if len(Da) == 10:
                break

    return Da  # [index, index, ...]


# 卡尔曼滤波生成高可用数据
def generatePT(Total_tra, Da):
    PT = []

    for da in Da:
        Lat = Total_tra[da][0]
        Lng = Total_tra[da][1]

        kf_trace = kf_trajectory(Lng, Lat)
        PT.append(kf_trace)

    return PT


def get_PT_Da_List(Total_tra, thred, labels):
    PT_list = []
    Da_list = []
    for t, trajectory in tqdm(enumerate(Total_tra), desc="开始计算Da和PT"):
        kind = labels[t]
        Da = filterByEuclidean(t, Total_tra, thred, kind, labels)
        PT = generatePT(Total_tra, Da)
        PT_list.append(PT)
        Da_list.append(Da)

    return PT_list, Da_list


# 获得PTP
def generatePTP(Total_tra, PT_list, Da_list, uio):
    PTP_list = []

    for t, trajectory in tqdm(enumerate(Total_tra), desc="开始计算PTP"):
        PTP = []
        Lat_a = trajectory[0]
        Lng_a = trajectory[1]

        PT_a = PT_list[t]

        for i, j in enumerate(Da_list[t]):
            if j != t:
                PT_j = PT_list[j]
                Lat_j = Total_tra[j][0]
                Lng_j = Total_tra[j][1]
                sim_aj = sim(Lat_a, Lng_a, Lat_j, Lng_j, PT_a, PT_j)

                if sim_aj < uio:
                    PTP.append(PT_a[i])
            else:
                PTP.append(PT_a[i])
        PTP_list.append(PTP)

    return PTP_list


# 计算每条轨迹的灵敏度（api请求有额度限制，废弃）
# def calculate_GF1(Total_tra, thred, uio, nums, url=my_url, ak=my_ak, POIs=POIs):
#     num_clusters, labels = Kmean_index(Total_tra, nums=nums)
#     PT_list, Da_list = get_PT_Da_List(Total_tra, thred, labels)
#     PTP_list = generatePTP(Total_tra, PT_list, Da_list, uio)
#
#     GF_list = []
#     for t, PTP in tqdm(enumerate(PTP_list), desc="开始计算GF"):
#         poiSet = set()
#         for trace in PTP:
#             for i in range(len(trace)):
#
#                 # 查询每个位置所属的POI
#                 position = "{},{}".format(trace[i][1], trace[i][0])
#                 left = 0
#                 right = 10
#                 while right <= len(POIs):
#                     query = "$".join(POIs[left:right])
#
#                     params = {
#                         "query": query,
#                         "location": position,
#                         "radius": "50",
#                         "output": "json",
#                         "ak": ak,
#                         "scope": 2
#                     }
#                     response = requests.get(url=url, params=params)
#                     if response:
#                         data = response.json()
#                         places = data["results"]
#                         if len(places) > 1:
#                             label = places[0]['detail_info']['label']
#                             print(label)
#                             poiSet.add(label)
#                             break
#         GF = len(poiSet)
#         GF_list.append(GF)
#     return GF_list


# 生成兴趣点（模拟真实场景）
def create_POI(cell_nums, db, POIs=POIs):
    # 设置随机种子
    random.seed(52)  # 可以选择任何整数作为种子
    # 将兴趣点和权重分开
    if db == "Geolife":
        interest_points = list(POIs.keys())
        ratings = list(POIs.values())
    else:
        # Shanghai语义集较小
        interest_points = list(POIs.keys())[0:50]
        ratings = list(POIs.values())[0:50]

    # 初始化结果列表
    result_list = []

    for _ in range(cell_nums):
        # 随机选择一个兴趣点，概率与评分成正比
        selected_interest = random.choices(interest_points, weights=ratings, k=1)[0]
        result_list.append(selected_interest)

    return result_list


# 计算灵敏度
def calculate_GF(Total_tra,
                 thred,
                 uio,
                 nums,
                 KindsOfArea,
                 db
                 ):

    alist, blist, slist, indexcell = KindsOfArea

    # 为每个网格生成POI
    POIList = create_POI(len(slist), db)

    num_clusters, labels = Kmean_index(Total_tra, nums=nums)
    PT_list, Da_list = get_PT_Da_List(Total_tra, thred, labels)

    # 打印信息
    # dalist = []
    # for Da in Da_list:
    #     dalist.append(len(Da))
    # print(dalist)

    PTP_list = generatePTP(Total_tra, PT_list, Da_list, uio)

    # 打印信息
    # plist = []
    # for PTP in PTP_list:
    #     plist.append(len(PTP))
    # print(plist)

    GF_list = []
    for t, PTP in tqdm(enumerate(PTP_list), desc="开始计算GF"):
        poiSet = set()
        for trace in PTP:
            for i in range(len(trace)):
                lng = trace[i][0]
                lat = trace[i][1]
                _, k = find_pointarea(lng, lat, alist, blist, slist, indexcell)
                if k != -1:
                    poiSet.add(POIList[k])

        GF = len(poiSet)
        GF_list.append(GF)

    return GF_list


def load_tradata_tcpp(tra_path):
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

        slng = min(lng_list)
        elng = max(lng_list)
        slat = min(lat_list)
        elat = max(lat_list)

        vorPoints, belongVoronoi = getvoronoi(lng_list, lat_list, slng, elng, slat, elat)
        Total_tra.append([lat_list, lng_list, belongVoronoi, vorPoints])

    return Total_tra, fnames  # [[纬度，经度，索引，该轨迹对应的Voronoi网格]...]


# 隐私预算分配
def addnoise_tcpp(Total_tra,     # 所有轨迹信息
                  δ,             # 隐私预算 δ
                  fnames,        # 轨迹文件名称
                  n,             # 敏感位置个数n
                  win,           # 滑动窗口的大小
                  database,      # 数据集名称
                  GF_list        # 灵敏度列表
                  ):

    change_path = "/Noise_Tra/TCPP/" + database + "/TCPP_e_{}_win_{}n_{}/".format(δ, win, n)
    total_save_path = output_path + change_path
    if not os.path.exists(total_save_path):
        # 如果不存在，则新建文件夹
        os.makedirs(total_save_path)

    # 遍历每条轨迹
    for t, trajectory in tqdm(enumerate(Total_tra), desc="轨迹加噪开始"):
        latlng_Noise = []
        Lat = trajectory[0]
        Lng = trajectory[1]
        belongVoronoi = trajectory[2]
        vorPoints = trajectory[3]
        pl = [random.uniform(0, 1) for _ in range(n)]   # 敏感位置对应的隐私级别
        weight = [random.uniform(0, 1) for _ in range(n)]    # 敏感位置的权重值
        length = len(belongVoronoi)

        GF = GF_list[t]

        newPLs = []    # 轨迹中每个位置的隐私等级
        rangeSet = {}
        # 预遍历每个位置点，计算相应权重
        for i in range(0, length):
            region_index = belongVoronoi[i]    # 获取该位置点所在的区域索引
            slpos = vorPoints[region_index]    # 区域的中心点（敏感位置）
            # 计算该位置点到敏感位置的距离
            dis = get_distance(Lng[i], Lat[i], slpos[0], slpos[1])
            newpl = pl[region_index] * weight[region_index] * (1/dis)
            newPLs.append(newpl)
            if rangeSet.get(region_index) is not None:
                rangeSet[region_index] += 1 / dis
            else:
                rangeSet[region_index] = 1 / dis

        sumpl = 0
        for i in range(0, length):
            region_index = belongVoronoi[i]
            newPLs[i] /= rangeSet[region_index]
            sumpl += newPLs[i]

        # k的倒数，因为负相关
        klist = [sumpl / newPLs[i] for i in range(0, length)]

        left = 0
        right = min(win, length)
        while left < length:
            ksum = sum(klist[left:right])
            for i in range(left, right):
                w = klist[i] / ksum    # 计算权重
                epsilon = δ * w

                lng_dis = np.random.laplace(loc=0, scale=GF / epsilon)
                lat_dis = np.random.laplace(loc=0, scale=GF / epsilon)

                tan_x = lat_dis / lng_dis

                # 计算反正切，得到弧度（以π为单位）
                radians = math.atan(tan_x)
                """
                    异常处理，会出现一些严重偏离范围的异常值
                """
                r = min(np.sqrt(lng_dis**2+lat_dis**2)*5e-2, 150)
                # print(r)

                lng_noise, lat_noise = addVectorToPos(Lng[i], Lat[i], r, radians)

                latlng_Noise.append([lat_noise, lng_noise])

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





