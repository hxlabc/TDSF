import os

from cor_martix import find_pointarea

os.environ["OMP_NUM_THREADS"] = '1'
import logging

# 获取根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.ERROR)  # 设置日志级别为 ERROR，以忽略低于此级别的消息

import random
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
from common import *
from Celldivide.utils import get_distance, perturb


# 加载轨迹数据集
def load_tradata_ntpp(tra_path):
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


# 生成一条轨迹的相似轨迹，作为历史轨迹
def create_history_tra(lat_list, lng_list, dis=1e-3):
    history_lat = []
    history_lng = []
    for i in range(len(lat_list)):
        disx = random.uniform(-dis, dis)
        disy = random.uniform(-dis, dis)
        history_lng.append(lng_list[i] + disy)
        history_lat.append(lat_list[i] + disx)
    return history_lat, history_lng


# 用HMM隐马尔可夫模型生成一条预测轨迹
def create_predict_tra(lat_list, lng_list):
    predict_lat = lat_list[:2]  # 假设第一个点是已知的
    predict_lng = lng_list[:2]
    length = len(lat_list)
    history_lat, history_lng = create_history_tra(lat_list, lng_list)
    n_components = min(length, 5)

    # 作差
    diff_lat = np.diff(history_lat)
    diff_lng = np.diff(history_lng)

    # 从第二个间隙开始遍历
    for i in range(2, length):
        # 从历史轨迹开始逐渐替换成真实位置点
        diff_lat[i - 2] = lat_list[i - 1] - lat_list[i - 2]
        diff_lng[i - 2] = lng_list[i - 1] - lng_list[i - 2]
        while True:
            try:
                model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000, min_covar=0.1)
                train_tra = np.column_stack([diff_lng, diff_lat])
                model.fit(train_tra)
                expected_returns_volumes = np.dot(model.transmat_, model.means_)

                x_pre = [[lng_list[i] - lng_list[i - 1], lat_list[i] - lat_list[i - 1]]]
                hidden_states = model.predict(x_pre)
                expected_returns = expected_returns_volumes[hidden_states][0]

                p_lng = lng_list[i - 1] + expected_returns[0]
                p_lat = lat_list[i - 1] + expected_returns[1]

                dis = get_distance(lng_list[i], lat_list[i], p_lng, p_lat)

                if dis > 150:
                    p_lng = lng_list[i] + (p_lng - lng_list[i]) * random.randint(50, 150) / dis
                    p_lat = lat_list[i] + (p_lat - lat_list[i]) * random.randint(50, 150) / dis

                # expected_returns_volumes[hidden_states] = [pre_lng, pre_lat]
                predict_lng.append(p_lng)
                predict_lat.append(p_lat)
                break
            except:
                # print("error!!!")
                pass

    return predict_lat, predict_lng


# 保存预测轨迹
def save_predict_tra(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Total_tra, fnames = load_tradata_ntpp(path)
    for t, trajectory in tqdm(enumerate(Total_tra)):
        Lat = trajectory[0]
        Lng = trajectory[1]

        p_Lat, p_Lng = create_predict_tra(Lat, Lng)

        latlng_Pre = np.column_stack([p_Lat, p_Lng])
        save_tra_path = save_path + fnames[t]
        with open(save_tra_path, 'w+') as f:
            for sublist in latlng_Pre:
                # 将子列表转换为字符串形式
                sublist_str = ','.join(map(str, sublist))
                # 写入文件
                f.write(sublist_str + '\n')
        f.close()


# 计算两条直线的夹角, 判断其是否为关键点
def is_keypoint(pos_l,  # 上一个点
                pos,  # 需要判断的点
                pos_r  # 下一个点
                ):
    p1 = np.array(pos_l)
    p2 = np.array(pos)
    p3 = np.array(pos_r)

    # 计算方向向量
    v1 = p2 - p1
    v2 = p3 - p2

    # 计算点积
    dot_product = np.dot(v1, v2)

    # 计算向量的模长
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 计算 cos(θ)
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # 由于浮点运算误差，cos_theta 的值可能略微超过范围 ([-1, 1])，可以使用 np.clip 函数限制其范围
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算夹角（以弧度为单位）
    theta_radians = np.arccos(cos_theta)

    # 判断其是否为关键点
    cos_theta = abs(cos_theta) if np.pi / 2 <= theta_radians <= np.pi else 0

    return cos_theta


# 给轨迹加噪
def addnoise_ntpp(Total_tra,  # 所有轨迹信息
                  Pre_tra,  # 预测轨迹
                  cor_message,  # 相关性矩阵及网格信息
                  δ,  # 隐私预算 δ
                  fnames,  # 轨迹文件名称
                  win,  # 滑动窗口的大小
                  β1,
                  β2,
                  Δε,
                  database  # 数据集名称
                  ):
    change_path = "/Noise_Tra/NTPP/" + database + "/NTPP_e_{}_win_{}/".format(δ, win)
    total_save_path = output_path + change_path
    if not os.path.exists(total_save_path):
        # 如果不存在，则新建文件夹
        os.makedirs(total_save_path)

    # 网格和相关性矩阵
    indexcell, alist, blist, slist, cor_mat = cor_message

    # 遍历每条轨迹
    for t, trajectory in tqdm(enumerate(Total_tra), desc="轨迹加噪开始"):
        latlng_Noise = []
        Lat = trajectory[0]
        Lng = trajectory[1]
        length = len(Lat)
        # print(length)

        # 加载对应的预测轨迹
        pre_trajectory = Pre_tra[t]
        pre_Lat = pre_trajectory[0]
        pre_Lng = pre_trajectory[1]

        left = 0
        right = min(win, length)
        while left < length:
            e_list = []  # 暂存一个滑动窗口内的隐私预算
            εr = δ / win
            # print("*********************")
            for i in range(left, right):
                εmax = δ - sum(e_list)

                if i != 0 and i != length - 1:
                    pos_l = [Lng[i - 1], Lat[i - 1]]
                    pos = [Lng[i], Lat[i]]
                    pos_r = [Lng[i + 1], Lat[i + 1]]
                    I = is_keypoint(pos_l, pos, pos_r)
                else:
                    I = 0

                """
                    dpo可能要调整，不然太大了 (已调整)
                """
                dpo = get_distance(Lng[i], Lat[i], pre_Lng[i], pre_Lat[i]) / 100
                PP = 1 / (1 + dpo)
                λ = β1 * PP - β2 * I
                εT = min(εr - λ * Δε, εmax)

                # 防止异常值导致报错
                if εT <= 0:
                    εT = 1e-2

                e_list.append(εT)
                DS = []
                barea, _ = find_pointarea(Lng[i], Lat[i], alist, blist, slist, indexcell)
                for _ in range(5):
                    lng_noise, lat_noise = perturb(εT, Lng[i], Lat[i])
                    # 防止加噪后出现nan的情况
                    if np.isnan(lng_noise) or np.isnan(lat_noise):
                        # print("nan ! ! !")
                        lng_noise = Lng[i] + random.uniform(-5e-4, 5e-4)
                        lat_noise = Lat[i] + random.uniform(-5e-4, 5e-4)

                    narea, _ = find_pointarea(lng_noise, lat_noise, alist, blist, slist, indexcell)
                    """
                        套用我的转移概率计算方式
                        如果扰动点和真实点在同一个网格，则考虑距离
                        否则考虑转移概率
                        取最小的zf  值为转移概率或者归一化距离
                    """
                    if barea is None or narea is None or barea.No == narea.No:  # 为空则记录距离
                        zf = get_distance(Lng[i], Lat[i], lng_noise, lat_noise) * 1e-4
                    else:
                        if cor_mat[narea.No].get(barea.No) is not None:
                            zf = cor_mat[narea.No][barea.No]
                        else:
                            zf = get_distance(Lng[i], Lat[i], lng_noise, lat_noise) * 1e-4

                    DS.append([lng_noise, lat_noise, zf])

                DS = sorted(DS, key=lambda x: x[2])
                lng_noise, lat_noise, _ = DS[0]  # 获取zf最小的作为扰动点
                latlng_Noise.append([lat_noise, lng_noise])

            # print(e_list)    # 测试隐私预算是否合理
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

