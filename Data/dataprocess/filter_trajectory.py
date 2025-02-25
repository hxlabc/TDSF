import os

import numpy as np
import xlwt

from Celldivide.draw import draw_distribution
from Celldivide.utils import get_distance
from common import Total_path


# 异常数据处理: 过滤那种只有一个点的轨迹
def dis_between_points(tra_path, del_flag=False):
    trafiles = os.scandir(tra_path)
    dis_list = []
    for tra_item in trafiles:
        dis = 0
        txtpath = tra_path + '/' + tra_item.name
        with open(txtpath, 'r+') as fp:
            lines = fp.readlines()
            length = len(lines)
            item1 = lines[0]
            item_list1 = item1.split(',')
            lat_p = float(item_list1[0])
            lng_p = float(item_list1[1])
            for item in lines[1:]:
                item_list = item.split(',')
                lat_i = float(item_list[0])
                lng_i = float(item_list[1])

                dis += get_distance(lng_i, lat_i, lng_p, lat_p)
                lng_p = lng_i
                lat_p = lat_i
        dis /= length-1
        if dis == 0:
            if del_flag:
                os.remove(txtpath)
                print(f"Deleted: {txtpath}")
        else:
            dis_list.append(dis)

    # print(len(dis_list))
    return np.mean(dis_list)


# 画轨迹长度分布图，并去除掉较短或较长的轨迹
def tra_filter(tra_path, dataBase, minlen=6, maxlen=500, name=""):
    clist = []
    for file_name in os.listdir(tra_path):
        if file_name.endswith('.txt'):  # 只处理txt文件
            file_path = os.path.join(tra_path, file_name)

            # 读取文件内容
            with open(file_path, 'r') as file:
                lines = file.readlines()

            if maxlen >= len(lines) >= minlen:
                clist.append(len(lines))
            else:
                os.remove(file_path)
                print(f"Deleted: {file_name}")

    print("过滤后轨迹数: ", len(clist))
    if name != "":
        draw_distribution(clist, 20, name, dataBase)


# 伯努利采样  p为采样概率
def bernoulli_sampling(p, size, lat, lng):
    lat_new = []
    lng_new = []

    # 使用numpy的random.binomial函数进行伯努利实验
    mask = np.random.binomial(n=1, p=p, size=size)

    for i in range(size):
        if mask[i] == 1:
            lat_new.append(lat[i])
            lng_new.append(lng[i])

    datalen = len(lat_new)

    return datalen, lat_new, lng_new


# 将采样后的数据保存到exel表中
def list2exel(lat, lng, Name):
    workbook = xlwt.Workbook(encoding='utf-8')  # 设置一个workbook，其编码是utf-8
    worksheet = workbook.add_sheet("lat_lng")  # 新增一个sheet
    worksheet.write(0, 0, label='经度')
    worksheet.write(0, 1, label='纬度')
    for i in range(len(lat)):  # 循环将a和b列表的数据插入至excel
        worksheet.write(i + 1, 0, label=lng[i])
        worksheet.write(i + 1, 1, label=lat[i])
    workbook.save(Total_path + "/Data/{}.xls".format(Name))  # 这里save需要特别注意，文件格式只能是xls，不能是xlsx，不然会报错

