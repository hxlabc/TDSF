import os

from tqdm import tqdm

from cor_martix import find_pointarea

os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import KMeans

from My_worker import Area
from Celldivide.utils import get_m1, get_m2
from common import *


# K-mean算法获取分割点
def divByKmean(k,  # 分类数目
               lng,  # 经度坐标集  type: list
               lat,  # 纬度坐标集    type: list
               region  # 区域范围
               ):
    lat_min, lat_max, lng_min, lng_max = region
    lng = np.array(lng).reshape(-1, 1)
    lat = np.array(lat).reshape(-1, 1)
    Km_lng = KMeans(n_clusters=k, random_state=0, n_init=10)
    Km_lat = KMeans(n_clusters=k, random_state=0, n_init=10)

    # 进行聚类
    Km_lng.fit(lng)
    Km_lat.fit(lat)

    # 获取每个簇的位置
    lng_cluster_labels = Km_lng.labels_
    lat_cluster_labels = Km_lat.labels_

    # 根据每个簇的最小值和最大值获取边界
    lng_div = []
    lat_div = []
    for i in range(k):
        lng_cluster_values = lng[lng_cluster_labels == i]
        lng_div.append(np.min(lng_cluster_values))

        lat_cluster_values = lat[lat_cluster_labels == i]
        lat_div.append(np.min(lat_cluster_values))

    # 对分割点进行排序并更换最小值，插入最大值
    lng_div = sorted(lng_div) + [lng_max]
    lat_div = sorted(lat_div) + [lat_max]
    lng_div[0] = lng_min
    lat_div[0] = lat_min

    # 返回分割点集合   type: list
    return lng_div, lat_div


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    方法1: 均匀AG划分
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def CellbyAvg(xlng,  # 经度坐标集
              ylat,  # 纬度坐标集
              Num,
              epsilon,  # 采样后的隐私预算
              B,  # 隐私预算分割比例 —— 用于一二级网格
              region,  # 区域范围
              thred_num01,  # 大于该人数才分割
              ):
    # 开始执行AG算法
    epsilon1 = B * epsilon  # 第一部分隐私预算
    epsilon2 = (1 - B) * epsilon  # 第二部分隐私预算

    Area02s = []  # 第一级网格集合
    Area03s = []  # 第二级网格集合

    real_num1 = 0
    real_num2 = 0

    vaild_num1 = 0  # 第一层有效网格数，即网格中工作者人数大于0
    vaild_num2 = 0  # 第二层有效网格数，即网格中工作者人数大于0

    m1 = get_m1(Num, epsilon)
    lat_min, lat_max, lng_min, lng_max = region

    # 计算 m1 * m1 个方格中每个方格的长宽
    disx = (lng_max - lng_min) / m1
    disy = (lat_max - lat_min) / m1

    lng_div = [lng_min] + [lng_min + u * disx for u in range(1, m1)] + [lng_max]
    lat_div = [lat_min] + [lat_min + u * disy for u in range(1, m1)] + [lat_max]

    # 第一层网格的划分
    for i in range(m1):
        for j in range(m1):
            x_up = lng_min + j * disx
            y_up = lat_min + i * disy + disy
            x_down = lng_min + j * disx + disx
            y_down = lat_min + i * disy

            area02 = Area([x_up, y_up], [x_down, y_down], 1)
            # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
            xlng, ylat = area02.divide(xlng, ylat)  # 更新x 和 y

            # 判断其是否为有效网格
            if area02.N > 0:
                vaild_num1 += 1

            area02.No = real_num1  # 记录该一级网格在Area02中的标号
            real_num1 += 1
            Area02s.append(area02)

    # 找第一层网格的邻居
    find_neighbor(Area02s, name="开始找一级网格邻居")

    for area02 in Area02s:
        flag = 1
        if area02.N > 0:
            e = epsilon1 / vaild_num1  # 计算Ai分配的隐私预算
            area02.add_noise(e, 1, vaild_num1)  # 添加噪声

            # 第二层网格的划分
            if area02.N_noise > thred_num01:  # 如果加噪后的人数少于等于0，直接跳过
                m2 = get_m2(area02.N_noise, epsilon2)

                # 计算 m2 * m2 个方格中每个方格的长宽
                disx = (area02.pos_down[0] - area02.pos_up[0]) / m2
                disy = (area02.pos_up[1] - area02.pos_down[1]) / m2
                xlng2 = area02.x
                ylat2 = area02.y

                flag = 0  # 用来标记未被划分的二级网格

                for i in range(m2):
                    for j in range(m2):
                        x_up = area02.pos_up[0] + j * disx
                        y_up = area02.pos_down[1] + i * disy + disy
                        x_down = area02.pos_up[0] + j * disx + disx
                        y_down = area02.pos_down[1] + i * disy

                        area03 = Area([x_up, y_up], [x_down, y_down], 2)
                        # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
                        xlng2, ylat2 = area03.divide(xlng2, ylat2)  # 更新x 和 y

                        # 判断其是否为有效网格
                        if area03.N > 0:
                            vaild_num2 += 1

                        area03.No = real_num2  # 记录该二级网格在B_area中的标号
                        area02.include_parts.append(area03.No)  # 将该二级网格的标号归类到该一级网格中
                        real_num2 += 1
                        Area03s.append(area03)

        # 将未被划分的一级网格列入二级网格，防止find_pointarea返回空
        if flag:
            # temp = area02.__copy__()
            # temp.level = 2
            # temp.No = real_num2
            temp = Area(area02.pos_up, area02.pos_down, 2)
            temp.No = real_num2
            area02.include_parts.append(temp.No)
            Area03s.append(temp)
            real_num2 += 1

    for area03 in Area03s:
        if area03.N > 0:
            e2 = epsilon2 / vaild_num2  # 计算Bi分配的隐私预算
            area03.add_noise(e2, 1, vaild_num2)  # 添加噪声

    # 找第二层网格的邻居，第一层网格作为索引
    find_neighbor(Area03s, Area02s, name="开始找二级网格邻居")

    return Area02s, Area03s, lng_div, lat_div


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    方法2: 一二层网格均用Kmean
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def Cell2KMean(xlng,  # 经度坐标集
               ylat,  # 纬度坐标集
               Num,
               epsilon,  # 采样后的隐私预算
               B,  # 隐私预算分割比例 —— 用于一二级网格
               region,  # 区域范围
               thred_num01,  # 大于该人数才分割
               kmean_num=500  # 一级大于该人数才用Kmean
               ):
    # 开始执行AG算法
    epsilon1 = B * epsilon  # 第一部分隐私预算
    epsilon2 = (1 - B) * epsilon  # 第二部分隐私预算

    Area02s = []  # 第一级网格集合
    Area03s = []  # 第二级网格集合

    real_num1 = 0
    real_num2 = 0

    vaild_num1 = 0  # 第一层有效网格数，即网格中工作者人数大于0
    vaild_num2 = 0  # 第二层有效网格数，即网格中工作者人数大于0

    m1 = get_m1(Num, epsilon)

    # 根据 m1 获取分割点集, 而不是均匀划分网格
    lng_div, lat_div = divByKmean(m1, xlng, ylat, region)

    # 第一层网格的划分
    for i in range(1, len(lng_div)):
        for j in range(1, len(lat_div)):
            area02 = Area([lng_div[i - 1], lat_div[j]], [lng_div[i], lat_div[j - 1]], 1)
            # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
            xlng, ylat = area02.divide(xlng, ylat)  # 更新x 和 y

            # 判断其是否为有效网格
            if area02.N > 0:
                vaild_num1 += 1

            area02.No = real_num1  # 记录该一级网格在Area02中的标号
            real_num1 += 1
            Area02s.append(area02)

    # 找第一层网格的邻居
    find_neighbor(Area02s, name="开始找一级网格邻居")

    for area02 in Area02s:
        flag = 1
        if area02.N > 0:
            e = epsilon1 / vaild_num1  # 计算Ai分配的隐私预算
            area02.add_noise(e, 1, vaild_num1)  # 添加噪声

            # 第二层网格的划分
            if area02.N_noise > thred_num01:  # 如果加噪后的人数少于等于0，直接跳过
                m2 = get_m2(area02.N_noise, epsilon2)

                # 计算 m2 * m2 个方格中每个方格的长宽
                disx = (area02.pos_down[0] - area02.pos_up[0]) / m2
                disy = (area02.pos_up[1] - area02.pos_down[1]) / m2
                xlng2 = area02.x
                ylat2 = area02.y

                flag = 0  # 用来标记未被划分的二级网格

                if area02.N < kmean_num:  # 均匀划分
                    for i in range(m2):
                        for j in range(m2):
                            x_up = area02.pos_up[0] + j * disx
                            y_up = area02.pos_down[1] + i * disy + disy
                            x_down = area02.pos_up[0] + j * disx + disx
                            y_down = area02.pos_down[1] + i * disy

                            area03 = Area([x_up, y_up], [x_down, y_down], 2)
                            # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
                            xlng2, ylat2 = area03.divide(xlng2, ylat2)  # 更新x 和 y

                            # 判断其是否为有效网格
                            if area03.N > 0:
                                vaild_num2 += 1

                            area03.No = real_num2  # 记录该二级网格在B_area中的标号
                            area02.include_parts.append(area03.No)  # 将该二级网格的标号归类到该一级网格中
                            real_num2 += 1
                            Area03s.append(area03)

                else:  # K-mean划分
                    print("2Kmean ! ! !")
                    # 根据 m2 获取分割点集, 而不是均匀划分网格
                    lng_div2, lat_div2 = divByKmean(m2, xlng2, ylat2, region)

                    for i in range(1, len(lng_div2)):
                        for j in range(1, len(lat_div2)):
                            area03 = Area([lng_div2[i - 1], lat_div2[j]], [lng_div2[i], lat_div2[j - 1]], 2)
                            # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
                            xlng2, ylat2 = area03.divide(xlng2, ylat2)  # 更新x 和 y

                            # 判断其是否为有效网格
                            if area03.N > 0:
                                vaild_num2 += 1

                            area03.No = real_num2  # 记录该二级网格在B_area中的标号
                            area02.include_parts.append(area03.No)  # 将该二级网格的标号归类到该一级网格中
                            real_num2 += 1
                            Area03s.append(area03)

        # 将未被划分的一级网格列入二级网格，防止find_pointarea返回空
        if flag:
            # temp = area02.__copy__()
            # temp.level = 2
            # temp.No = real_num2
            temp = Area(area02.pos_up, area02.pos_down, 2)
            temp.No = real_num2
            area02.include_parts.append(temp.No)
            Area03s.append(temp)
            real_num2 += 1

    for area03 in Area03s:
        if area03.N > 0:
            e2 = epsilon2 / vaild_num2  # 计算Bi分配的隐私预算
            area03.add_noise(e2, 1, vaild_num2)  # 添加噪声

    find_neighbor(Area03s, Area02s, name="开始找二级网格邻居")

    return Area02s, Area03s, lng_div, lat_div


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    方法3: 只有第一层网格用Kmean
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# 分割第一层网格和第二层网格
def Cell1KMean(xlng,  # 经度坐标集
               ylat,  # 纬度坐标集
               Num,
               epsilon,  # 采样后的隐私预算
               B,  # 隐私预算分割比例 —— 用于一二级网格
               region,  # 区域范围
               thred_num01
               ):
    # 开始执行AG算法
    epsilon1 = B * epsilon  # 第一部分隐私预算
    epsilon2 = (1 - B) * epsilon  # 第二部分隐私预算

    Area02s = []  # 第一级网格集合
    Area03s = []  # 第二级网格集合

    real_num1 = 0
    real_num2 = 0

    vaild_num1 = 0  # 第一层有效网格数，即网格中工作者人数大于0
    vaild_num2 = 0  # 第二层有效网格数，即网格中工作者人数大于0

    m1 = get_m1(Num, epsilon)

    # 根据 m1 获取分割点集, 而不是均匀划分网格
    lng_div, lat_div = divByKmean(m1, xlng, ylat, region)

    # 第一层网格的划分
    for i in range(1, len(lng_div)):
        for j in range(1, len(lat_div)):
            area02 = Area([lng_div[i - 1], lat_div[j]], [lng_div[i], lat_div[j - 1]], 1)
            # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
            xlng, ylat = area02.divide(xlng, ylat)  # 更新x 和 y

            # 判断其是否为有效网格
            if area02.N > 0:
                vaild_num1 += 1

            area02.No = real_num1  # 记录该一级网格在Area02中的标号
            real_num1 += 1
            Area02s.append(area02)

    # 找第一层网格的邻居
    find_neighbor(Area02s, name="开始找一级网格邻居")

    for area02 in Area02s:
        flag = 1
        if area02.N > 0:
            e = epsilon1 / vaild_num1  # 计算Ai分配的隐私预算
            area02.add_noise(e, 1, vaild_num1)  # 添加噪声

            # 第二层网格的划分
            if area02.N_noise > thred_num01:  # 如果加噪后的人数少于等于5，直接跳过
                m2 = get_m2(area02.N_noise, epsilon2)

                # 计算 m2 * m2 个方格中每个方格的长宽
                disx = (area02.pos_down[0] - area02.pos_up[0]) / m2
                disy = (area02.pos_up[1] - area02.pos_down[1]) / m2
                xlng2 = area02.x
                ylat2 = area02.y

                flag = 0  # 用来标记未被划分的二级网格

                for i in range(m2):
                    for j in range(m2):
                        x_up = area02.pos_up[0] + j * disx
                        y_up = area02.pos_down[1] + i * disy + disy
                        x_down = area02.pos_up[0] + j * disx + disx
                        y_down = area02.pos_down[1] + i * disy

                        area03 = Area([x_up, y_up], [x_down, y_down], 2)
                        # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
                        xlng2, ylat2 = area03.divide(xlng2, ylat2)  # 更新x 和 y

                        # 判断其是否为有效网格
                        if area03.N > 0:
                            vaild_num2 += 1

                        area03.No = real_num2  # 记录该二级网格在B_area中的标号
                        area02.include_parts.append(area03.No)  # 将该二级网格的标号归类到该一级网格中
                        real_num2 += 1
                        Area03s.append(area03)

        # 将未被划分的一级网格列入二级网格，防止find_pointarea返回空
        if flag:
            # temp = area02.__copy__()
            # temp.level = 2
            # temp.No = real_num2
            temp = Area(area02.pos_up, area02.pos_down, 2)
            temp.No = real_num2
            area02.include_parts.append(temp.No)
            Area03s.append(temp)
            real_num2 += 1

    for area03 in Area03s:
        if area03.N > 0:
            e2 = epsilon2 / vaild_num2  # 计算Bi分配的隐私预算
            area03.add_noise(e2, 1, vaild_num2)  # 添加噪声

    # 找第二层网格的邻居，第一层网格作为索引
    # find_neighbor(Area03s, Area02s, name="开始找二级网格邻居")

    return Area02s, Area03s, lng_div, lat_div


# 判断该网格是否在GR_max内
def is_inGRmax(area, GR_max):
    x1, y1 = area.pos_up
    x2, y2 = area.pos_down
    # 一个矩形的上下顶点都在GR_max内改矩形才在GR_max内
    return GR_max.is_inArea(x1, y1, True) and GR_max.is_inArea(x2, y2, True)


# 找相邻网格
def find_neighbor(X_area, blist=None, name=''):
    # 一级网格暴力查找邻居
    if blist is None:
        n = len(X_area)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if X_area[i].is_neighbor(X_area[j].pos_up, X_area[j].pos_down):
                        X_area[i].add_neighbor(j)
    # 二、三级网格通过二级网格索引加速查找邻居
    else:
        for b_area in tqdm(blist, desc=name):
            include_area = b_area.include_parts
            waitlist = include_area[:]
            for nei in b_area.neighbors:
                waitlist += blist[nei].include_parts
            for ia in include_area:
                for wl in waitlist:
                    if ia != wl:
                        if X_area[ia].is_neighbor(X_area[wl].pos_up, X_area[wl].pos_down):
                            X_area[ia].add_neighbor(wl)


# 获得一级网格的索引 m1 = 11 = 4+4+3    划分成3*3网格
def getindex(alist,
             lng_div,  # 分割点   type: list
             lat_div  # 分割点   type: list
             ):
    indexcell = []

    m = len(lng_div)
    new_lng_div = [lng_div[0], lng_div[m // 3], lng_div[m * 2 // 3], lng_div[m - 1]]
    new_lat_div = [lat_div[0], lat_div[m // 3], lat_div[m * 2 // 3], lat_div[m - 1]]

    num = 0
    # 索引网格的划分
    for i in range(1, len(new_lng_div)):
        for j in range(1, len(new_lat_div)):
            area01 = Area([new_lng_div[i - 1], new_lat_div[j]], [new_lng_div[i], new_lat_div[j - 1]], 0)
            area01.No = num
            num += 1
            for ant in alist:
                if is_inGRmax(ant, area01):
                    area01.include_parts.append(ant.No)
            indexcell.append(area01)

    return indexcell


# 进一步细化网格，三级网格
def divide_smaller_area(blist, thred_num02, thred_midu, min_area=900):
    slist = []
    count = 0
    for t, b_area in enumerate(blist):
        S = b_area.get_square(True)
        m3 = int(np.sqrt(round(S / min_area, 0)))
        if m3 == 0:
            m3 += 1
        # print(t, m3)
        p2 = b_area.N / S  # 计算网格密度

        if b_area.N >= thred_num02 and p2 > thred_midu:
            disx = (b_area.pos_down[0] - b_area.pos_up[0]) / m3
            disy = (b_area.pos_up[1] - b_area.pos_down[1]) / m3
            xlng = b_area.x
            ylat = b_area.y

            for i in range(m3):
                for j in range(m3):
                    x_up = b_area.pos_up[0] + j * disx
                    y_up = b_area.pos_down[1] + i * disy + disy
                    x_down = b_area.pos_up[0] + j * disx + disx
                    y_down = b_area.pos_down[1] + i * disy

                    n_area = Area([x_up, y_up], [x_down, y_down], 3)
                    # 将属于该区域的点放入该区域，并返回没有划分进入该区域点的集合，提高算法效率
                    xlng, ylat = n_area.divide(xlng, ylat)  # 更新x 和 y

                    n_area.No = count
                    count += 1
                    b_area.include_parts.append(n_area.No)
                    slist.append(n_area)

        else:
            n_area = Area([b_area.pos_up[0], b_area.pos_up[1]], [b_area.pos_down[0], b_area.pos_down[1]], 3)
            n_area.No = count
            n_area.N = b_area.N
            count += 1
            b_area.include_parts.append(n_area.No)
            slist.append(n_area)

    # find_neighbor(slist, blist, name="开始找三级网格邻居")
    return blist, slist


# 过滤网格
def filter_cell(slist, maxArea=22000):
    numf = 0  # 有效网格数
    nump = 0  # 人数
    sumArea = 0

    for snt in tqdm(slist, desc="过滤网格"):
        S = snt.get_square(True)
        if S < maxArea:
            sumArea += S
            numf += 1
            nump += snt.N
            snt.r = 1  # 标记为有效网格
        else:
            snt.r = 0  # 无效网格
    print("有效网格数：", numf)
    print("覆盖人数：", nump)
    print("平均网格面积：", round(sumArea / numf, 2))
    return slist

# 过滤网格
# def filter_cell(slist, maxArea=22000):
#     numf = 0  # 有效网格数
#     nump = 0  # 人数
#     sumArea = 0
#
#     for snt in tqdm(slist, desc="过滤网格"):
#         S = snt.get_square(True)
#         if S < maxArea:
#             ok = False
#             if snt.N > 0:    # 该网格里有人为有效网格
#                 ok = True
#             else:           # 该网格的邻居网格至少一个有人也为有效网格
#                 nei_list = snt.neighbors
#                 for nei in nei_list:
#                     if slist[nei].N > 0:
#                         ok = True
#                     # else:
#                     #     sub_nei_list = slist[nei].neighbors
#                     #     for sub_nei in sub_nei_list:
#                     #         if slist[sub_nei].N > 0:
#                     #             ok = True
#                     #             break
#                     if ok:
#                         break
#             if ok:
#                 sumArea += S
#                 numf += 1
#                 nump += snt.N
#                 snt.r = 1  # 标记为有效网格
#         else:
#             snt.r = 0  # 无效网格
#     print("有效网格数：", numf)
#     print("覆盖人数：", nump)
#     print("平均网格面积：", round(sumArea/numf, 2))
#     return slist


# def filter_cell2(Lat, Lng, alist, blist, slist, indexcell, maxArea=22000):
#     numf = 0  # 有效网格数
#     nump = 0  # 人数
#     sumArea = 0
#
#     datalen = len(Lat)
#
#     # 初始化全部为0
#     for i in range(len(slist)):
#         slist[i].r = 0
#
#     for i in tqdm(range(datalen), desc="过滤网格"):
#         _, k = find_pointarea(Lng[i], Lat[i], alist, blist, slist, indexcell)
#         if slist[k].r != 1:
#             S = slist[k].get_square(True)
#             if S < maxArea:
#                 sumArea += S
#                 numf += 1
#                 nump += slist[k].N
#                 slist[k].r = 1  # 标记为有效网格
#
#     print("有效网格数：", numf)
#     print("覆盖人数：", nump)
#     print("平均网格面积：", round(sumArea / numf, 2))
#     return slist

