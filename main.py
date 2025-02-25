"""
 各种轨迹加噪实验
"""
from Celldivide.draw import draw_high_point_rate
from Data.dataprocess.filter_trajectory import dis_between_points
from OtherMedthods.LPPD import load_tradata_lppd, addnoise_lppd
from OtherMedthods.NTPP import load_tradata_ntpp, addnoise_ntpp, save_predict_tra
from OtherMedthods.TCPP import addnoise_tcpp, load_tradata_tcpp, calculate_GF
from TDSF import addnoise2tra
from common import *
from cor_martix import loaddata_from_xls, load_tradata, get_cormatrix
from metric import load_message

# my_database = "Geolife"
# my_database = "Chengdu"
my_database = "Shanghai"

tra_path = Total_path + "/Data/{}_tra".format(my_database)
train_tra_path = output_path + "/Train_Tra/{}_train".format(my_database)


if my_database == "Geolife":
    path0 = geo_path0
    path1 = geo_path1
    path2 = geo_path2
    path3_list = geo_path3

    d_thred = geo_d_thred

    ntpp_rate = geo_ntpp_rate
    ntpp_B = geo_ntpp_B

elif my_database == "Chengdu":
    path0 = cd_path0
    path1 = cd_path1
    path2 = cd_path2
    path3_list = cd_path3

    d_thred = cd_d_thred

    ntpp_rate = cd_ntpp_rate
    ntpp_B = cd_ntpp_B
else:
    path0 = sh_path0
    path1 = sh_path1
    path2 = sh_path2
    path3_list = sh_path3

    d_thred = sh_d_thred

    ntpp_rate = sh_ntpp_rate
    ntpp_B = sh_ntpp_B


# 画轨迹高相关性点的比例图
def draw_hpr():
    # 加载网格信息
    indexcell = loaddata_from_xls(path0, 0)
    alist = loaddata_from_xls(path1, 1)
    blist = loaddata_from_xls(path2, 2)

    lo = 0
    slist = []
    for path3 in path3_list:
        slist += loaddata_from_xls(path3, 3, lo)
        lo = len(slist)

    # 加载训练轨迹信息，用来训练相关性矩阵
    Train_tra, _ = load_tradata(train_tra_path, alist, blist, slist, indexcell)
    # 加载轨迹信息，用来实验
    Total_tra, fnames = load_tradata(tra_path, alist, blist, slist, indexcell)

    # 加载相关性矩阵
    cor_mat = get_cormatrix(Train_tra, slist)

    # 画轨迹高相关性点的比例图
    draw_high_point_rate(Total_tra, cor_mat, my_database)


"""
  1) Our Method
"""


def add_noise_by_TDSF(epsilon, win):
    # 加载网格信息
    indexcell = loaddata_from_xls(path0, 0)
    alist = loaddata_from_xls(path1, 1)
    blist = loaddata_from_xls(path2, 2)

    lo = 0
    slist = []
    for path3 in path3_list:
        slist += loaddata_from_xls(path3, 3, lo)
        lo = len(slist)

    my_KindsOfArea = [alist, blist, slist, indexcell]

    # 加载训练轨迹信息，用来训练相关性矩阵
    Train_tra, _ = load_tradata(train_tra_path, alist, blist, slist, indexcell)
    # 加载轨迹信息，用来实验
    Total_tra, fnames = load_tradata(tra_path, alist, blist, slist, indexcell)

    # 加载相关性矩阵
    cor_mat = get_cormatrix(Train_tra, slist)

    for e in epsilon:
        addnoise2tra(Total_tra, cor_mat, e, my_KindsOfArea, win, fnames, my_database)
        print(str(e) + "finish ! ! !")


"""
  2) TCPP Method
"""


def add_noise_by_TCPP(epsilon, win, dis_thred, n=6):
    Total_tra, fnames = load_tradata_tcpp(tra_path)

    # 加载网格信息
    indexcell = loaddata_from_xls(path0, 0)
    alist = loaddata_from_xls(path1, 1)
    blist = loaddata_from_xls(path2, 2)

    lo = 0
    slist = []
    for path3 in path3_list:
        slist += loaddata_from_xls(path3, 3, lo)
        lo = len(slist)

    my_KindsOfArea = [alist, blist, slist, indexcell]

    GF_list = calculate_GF(Total_tra, dis_thred, 0.7, 20, my_KindsOfArea, my_database)

    for e in epsilon:
        addnoise_tcpp(Total_tra, e, fnames, n, win, my_database, GF_list)
        print(str(e) + "finish ! ! !")


"""
  3) NTPP Method
"""


def add_noise_by_NTPP(epsilon, win):
    pre_path = output_path + "/Train_Tra/NTPP_{}_pre/".format(my_database)
    # save_predict_tra(tra_path, pre_path)

    Total_tra, fnames = load_tradata_ntpp(tra_path)
    Pre_tra, _ = load_tradata_ntpp(pre_path)

    # 加载相关性矩阵
    cell_list = [path0, path1, path2, path3_list]
    indexcell = loaddata_from_xls(path0, 0)
    alist = loaddata_from_xls(path1, 1)
    blist = loaddata_from_xls(path2, 2)

    lo = 0
    slist = []
    for path3 in path3_list:
        slist += loaddata_from_xls(path3, 3, lo)
        lo = len(slist)

    cor_mat = load_message(train_tra_path, cell_list, "cor")
    cor_message = [indexcell, alist, blist, slist, cor_mat]

    # Δε大概设为隐私预算的1/5
    for e in epsilon:
        icre_epsilon = e / ntpp_rate
        addnoise_ntpp(Total_tra, Pre_tra, cor_message, e, fnames, win, ntpp_B, 1-ntpp_B, icre_epsilon, my_database)
        print(str(e) + "finish ! ! !")


"""
  4) LPPD Method
"""


def add_noise_by_LPPD(epsilon, win):
    Total_tra, fnames = load_tradata_lppd(tra_path)

    for e in epsilon:
        addnoise_lppd(Total_tra, e, fnames, win, 0.6, 0.4, 0.3, my_database)
        print(str(e) + "finish ! ! !")


if __name__ == '__main__':
    my_epsilon = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]

    my_win = 10

    # draw_hpr()

    add_noise_by_TDSF(my_epsilon, my_win)

    add_noise_by_TCPP(my_epsilon, my_win, d_thred)
    add_noise_by_LPPD(my_epsilon, my_win)
    add_noise_by_NTPP(my_epsilon, my_win)
