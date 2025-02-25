import numpy as np
import math
from haversine import haversine

from common import earth_radius, rad_of_deg, deg_of_rad


# 计算第一网格粒度
def get_m1(N, epsilon):
    c1 = 10
    res = np.sqrt((N * epsilon) / c1) / 4
    res = math.ceil(res)
    return max(10, res)


# 计算第二网格粒度
def get_m2(N_noise, epsilon):
    c2 = np.sqrt(2)
    res = np.sqrt((N_noise * epsilon) / c2)
    res = math.ceil(res)
    return res


# 计算标准差圆半径
def get_r(x, y):
    n = len(x)
    if n == 1 or n == 2:  # 特殊情况处理
        return 1
    if n > 2:  # 防止出现分母为0
        xn = np.mean(x)
        yn = np.mean(y)
        cnt = 0
        for i in range(n):
            dx = x[i] - xn
            dy = y[i] - yn
            cnt += dx ** 2 + dy ** 2
        cnt /= n - 2
        r = np.sqrt(cnt)
    else:
        return -1  # 返回值为-1表示该网格人数为0
    return r


# 计算两个经纬度坐标间的距离
def get_distance(lng1, lat1, lng2, lat2):
    lyon = [lat1, lng1]
    paris = [lat2, lng2]

    return haversine(lyon, paris) * 1000  # 单位米


# 工作者接受率
def worker_receive_rate(pmax, d, dmax):
    if dmax < d:
        return 0
    return pmax * (1 - d / dmax)


# 网格接受率
def area_receive_rate(n, p):
    return 1 - (1 - p) ** n


# 最小子网格面积
def min_area(p, p_gr, thres, N, square):
    S_min = -1

    if 0 <= p_gr < 1 and 0 < p < 1 and N > 0:  # 分母不能为0
        temp = (1 - thres) / (1 - p_gr)

        N_min = math.log(temp, 1 - p)
        S_min = (N_min / N) * square

    return S_min


# 根据分数计算每一个区间的可能性（指数机制）
def score_p(epsilon, sensitivity, score):
    probability = np.exp(epsilon * score / (2 * sensitivity))
    return probability


# 根据转移概率计算相关性的值
def tran2corr(tranpro, maxx):
    # k为修正参数，根据经过网格次数的最大值确定
    if maxx <= 10:
        k = 3
    elif maxx <= 50:
        k = 5
    elif maxx <= 100:
        k = 7
    elif maxx <= 150:
        k = 9
    elif maxx <= 250:
        k = 10
    else:
        k = 13
    return 1 - math.exp(-k * tranpro)


"""
    @ 地理不可区分加噪模块
"""


# 地理不可区分中LambertW函数的区间(−∞，−1)分支的定义
def LambertW(x):
    # Min diff decides when the while loop ends
    min_diff = 1e-10
    if x == -1 / np.e:
        return -1
    elif (x < 0) and (x > -1 / np.e):
        q = np.log(-x)
        p = 1
        while abs(p - q) > min_diff:
            p = (q * q + x / np.exp(q)) / (q + 1)
            q = (p * p + x / np.exp(p)) / (p + 1)
        # determine the precision of the float number to be returned
        return np.round(1000000 * q) / 1000000
    elif x == 0:
        return 0
    else:
        return 0


# 根据LambertW计算半径r
def inverseCumulativeGamma(epsilon, z):
    # x = (z - 1) / epsilon * 1e-2
    x = (z - 1) / epsilon
    res = LambertW(x)
    return (-(res + 1)) / epsilon


# 根据半径r和角度确定扰动点
def addVectorToPos(lng, lat, distance, angle) -> (int, int):
    '''
    params:
        int: lng - longitude
        int: lat - latitude
        int: distance - this is going to be the mechanism for perturbation?
        int: angle - the angle
    '''
    ang_distance = distance / earth_radius
    lat1 = rad_of_deg(lat)
    lng1 = rad_of_deg(lng)

    lat2 = np.arcsin(np.sin(lat1) * np.cos(ang_distance) + np.cos(lat1) * np.sin(ang_distance) * np.cos(angle))

    lng2 = lng1 + np.arctan2(np.sin(angle) * np.sin(ang_distance) * np.cos(lat1),
                             np.cos(ang_distance) - np.sin(lat1) * np.sin(lat2))
    lng2 = (lng2 + 3 * np.pi) % (2 * np.pi) - np.pi  # Normalize to -180 to +180

    latnew = deg_of_rad(lat2)
    lngnew = deg_of_rad(lng2)
    return lngnew, latnew


# 对单个点进行地理无法区分加噪
def perturb(epsilon, lng, lat):
    # Generating some random angle in [0, 2 * pi)
    theta = np.random.rand() * np.pi * 2

    # Generating some random number in [0, 1)
    z = np.random.rand()

    r = inverseCumulativeGamma(epsilon, z)

    return addVectorToPos(lng, lat, r, theta)


def sim(Lat_a, Lng_a, Lat_j, Lng_j, PT_a, PT_j):
    n = min(len(PT_a), len(PT_j))
    PT_a = PT_a[:n]
    PT_j = PT_j[:n]
    w = len(Lat_a)

    Sim_aj = 0
    for i in range(n):
        sumCv = 0
        for u in range(w):
            cov_aj = cov(u, Lat_a[u], Lng_a[u], Lat_j[u], Lng_j[u], PT_a, PT_j)
            var_aa = cov(u, Lat_a[u], Lng_a[u], Lat_a[u], Lng_a[u], PT_a, PT_a)
            var_jj = cov(u, Lat_j[u], Lng_j[u], Lat_j[u], Lng_j[u], PT_j, PT_j)

            sumCv += cov_aj / (np.sqrt(var_aa * var_jj))
        sumCv /= w
        Sim_aj += sumCv

    Sim_aj /= n

    # print(Sim_aj)
    return Sim_aj


def cov(u, lat_a, lng_a, lat_j, lng_j, PT_a, PT_j):
    n = len(PT_a)

    cov_aj = 0
    for i in range(n):
        # cov_aj += get_distance(lng_a, lat_a, PT_a[i][u][0], PT_a[i][u][1]) * get_distance(lng_j, lat_j, PT_j[i][u][0], PT_j[i][u][1])
        a = abs(lng_a-PT_a[i][u][0]) + abs(lat_a-PT_a[i][u][1])
        b = abs(lng_j-PT_j[i][u][0]) + abs(lat_j-PT_j[i][u][1])
        cov_aj += a*b

    if n > 1:
        cov_aj /= n-1
    else:
        cov_aj /= n

    return cov_aj
