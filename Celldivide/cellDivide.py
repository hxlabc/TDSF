import argparse


import pandas as pd
import xlwt

from Celldivide.AGmethod import Cell2KMean, divide_smaller_area, filter_cell, getindex, Cell1KMean, CellbyAvg
from Celldivide.draw import draw_areas, draw_filterareas, draw0and1
from Data.dataprocess.pro_Geolife import plt2list
from common import *


# 加载数据
def data_load(file_path):
    worker = pd.read_excel(file_path)

    lng = list(worker['经度'])
    lat = list(worker['纬度'])
    datalen = len(lat)

    return datalen, lat, lng


# 将网格信息存入xls
def saveCell2Exel(blist, fname, level, dataBase):
    workbook = xlwt.Workbook(encoding='utf-8')  # 设置一个workbook，其编码是utf-8
    worksheet = workbook.add_sheet("lat_lng")  # 新增一个sheet
    worksheet.write(0, 0, label='x_up')
    worksheet.write(0, 1, label='y_up')
    worksheet.write(0, 2, label='x_down')
    worksheet.write(0, 3, label='y_down')
    worksheet.write(0, 4, label='网格序号')
    worksheet.write(0, 5, label='层号')

    if level == 1 or level == 0:
        worksheet.write(0, 6, label='所包含网格编号')
    elif level == 2:
        worksheet.write(0, 6, label='所包含网格编号')
        worksheet.write(0, 7, label='邻居')
    else:  # level == 3
        worksheet.write(0, 6, label='邻居')
        worksheet.write(0, 7, label='有效标记')

    for i in range(len(blist)):
        x_up, y_up = blist[i].pos_up
        x_down, y_down = blist[i].pos_down
        no = blist[i].No
        lev = blist[i].level
        r = blist[i].r

        worksheet.write(i + 1, 0, label=x_up)
        worksheet.write(i + 1, 1, label=y_up)
        worksheet.write(i + 1, 2, label=x_down)
        worksheet.write(i + 1, 3, label=y_down)
        worksheet.write(i + 1, 4, label=no)
        worksheet.write(i + 1, 5, label=lev)

        if level == 1 or level == 0:
            worksheet.write(i + 1, 6, label=', '.join(str(c) for c in blist[i].include_parts))
        elif level == 2:
            worksheet.write(i + 1, 6, label=', '.join(str(c) for c in blist[i].include_parts))
            worksheet.write(i + 1, 7, label=', '.join(str(c) for c in blist[i].neighbors))
        else:  # level == 3
            worksheet.write(i + 1, 6, label=', '.join(str(c) for c in blist[i].neighbors))
            worksheet.write(i + 1, 7, label=r)
    workbook.save(Total_path + "/run/RUN_{}/".format(dataBase) + dataBase + fname + ".xls")


# 输入参数
def parse_opt():
    # db = "Geolife"
    # db = "Chengdu"
    db = "Shanghai"

    parser = argparse.ArgumentParser(description="Centralized Differential Privacy")
    parser.add_argument('--epsilon', type=float, default=1, help='Privacy budget')
    parser.add_argument('--B', type=float, default=0.5, help='Budget factor')
    parser.add_argument('--source', type=str, default=Total_path + '/Data/{}.xls'.format(db), help='Data path')
    parser.add_argument('--save', default=False, help='Whether to save the picture')
    parser.add_argument('--db', type=str, default=db, help='DataBase')

    opt = parser.parse_args()
    parser.print_help()
    print(opt)

    return opt


# 运行程序
def run(opt):
    epsilon = opt.epsilon  # 总共的隐私预算
    B = opt.B  # 预算因子
    file_path = opt.source  # 数据的路径
    db = opt.db

    if db == "Geolife":
        my_lat_min = geo_lat_min
        my_lat_max = geo_lat_max
        my_lng_min = geo_lng_min
        my_lng_max = geo_lng_max
        my_thred_num01 = geo_thred_num01
        my_max_area = geo_max_area
        my_thred_num02 = geo_thred_num02
        my_thred_midu = geo_thred_midu
    elif db == "Chengdu":
        my_lat_min = cd_lat_min
        my_lat_max = cd_lat_max
        my_lng_min = cd_lng_min
        my_lng_max = cd_lng_max
        my_thred_num01 = cd_thred_num01
        my_max_area = cd_max_area
        my_thred_num02 = cd_thred_num02
        my_thred_midu = cd_thred_midu
    else:
        my_lat_min = sh_lat_min
        my_lat_max = sh_lat_max
        my_lng_min = sh_lng_min
        my_lng_max = sh_lng_max
        my_thred_num01 = sh_thred_num01
        my_max_area = sh_max_area
        my_thred_num02 = sh_thred_num02
        my_thred_midu = sh_thred_midu

    # 加载数据 和 区域
    datalen, lat, lng = data_load(file_path)
    region = [my_lat_min, my_lat_max, my_lng_min, my_lng_max]

    # 加载一二级网格和分割点集
    """
        三种网格划分方法
    """
    # celldiv_method = "Avg"
    celldiv_method = "Kmean"
    # celldiv_method = "Kmean2"

    if celldiv_method == "Avg":
        alist, blist, lng_div, lat_div = CellbyAvg(lng, lat, datalen, epsilon, B, region, my_thred_num01)
    elif celldiv_method == "Kmean":
        alist, blist, lng_div, lat_div = Cell1KMean(lng, lat, datalen, epsilon, B, region, my_thred_num01)
    else:
        alist, blist, lng_div, lat_div = Cell2KMean(lng, lat, datalen, epsilon, B, region, my_thred_num01)

    # 加载三级网格并更新二级网格的包含
    blist, slist = divide_smaller_area(blist, my_thred_num02, my_thred_midu)

    # 加载索引网格
    indexcell = getindex(alist, lng_div, lat_div)

    # 过滤三级网格
    slist = filter_cell(slist, my_max_area)

    print("一级网格数量：", len(alist))
    print("二级网格数量：", len(blist))
    print("三级网格数量：", len(slist))

    # print("开始存储！！！")
    # saveCell2Exel(indexcell, "索引网格", 0, db)
    # saveCell2Exel(alist, "第一层网格", 1, db)
    # saveCell2Exel(blist, "第二层网格", 2, db)
    #
    # lo = 0
    # ro = min(len(slist), 60000)
    # numb = 1
    # while lo != ro:
    #     saveCell2Exel(slist[lo:ro], "第三层网格0{}".format(numb), 3, db)
    #     lo = ro
    #     ro = min(len(slist), ro + 60000)
    #     numb += 1

    # print("开始画图！！！")
    # draw_areas(alist, "一层网格分割{}".format(celldiv_method), region, db, lng, lat)
    # draw_areas(blist, "二层网格分割{}".format(celldiv_method), region, db, lng, lat)
    draw_areas(slist, "三层网格分割{}".format(celldiv_method), region, db)
    # draw_filterareas(slist, "有效网格可视{}".format(celldiv_method), region, db)
    # draw0and1(indexcell, alist, "索引网格示意图", region, db)


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)



