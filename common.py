# 项目地址
import numpy as np

Total_path = "D:/Different Privacy/MyWork/New DP Trajectory Correlation"

# 原始数据集地址
total_data_path = "D:/Different Privacy/MyWork/TraData"

# 外部存储地址
output_path = "D:/Different Privacy/MyWork/OutputTra"

earth_radius = 6.3781e6

# 相关性阈值
cor_thred = 0.4

# 接口地址
my_url = "https://api.map.baidu.com/place/v2/search"

# 此处填写你在控制台-应用管理-创建应用后获取的AK
my_ak = "tArgsLSJvDVIpoX0j4sxfWHpCn9gAVkv"


# Radians -> Degrees
def deg_of_rad(ang):
    return np.degrees(ang)


# Degrees -> Radians
def rad_of_deg(ang):
    return np.radians(ang)


"""
   1) Geolife 数据集范围
"""
geo_lat_min = 39.90
geo_lat_max = 40.025
geo_lng_min = 116.275
geo_lng_max = 116.475

geo_path0 = Total_path + "/run/RUN_Geolife/Geolife索引网格.xls"
geo_path1 = Total_path + "/run/RUN_Geolife/Geolife第一层网格.xls"
geo_path2 = Total_path + "/run/RUN_Geolife/Geolife第二层网格.xls"

geo_path3_1 = Total_path + "/run/RUN_Geolife/Geolife第三层网格01.xls"
geo_path3_2 = Total_path + "/run/RUN_Geolife/Geolife第三层网格02.xls"
geo_path3_3 = Total_path + "/run/RUN_Geolife/Geolife第三层网格03.xls"
geo_path3 = [geo_path3_1, geo_path3_2, geo_path3_3]

geo_thred_num01 = 5
geo_max_area = 22000
geo_thred_num02 = 0
geo_thred_midu = 5e-6

geo_d_thred = 200

geo_ntpp_rate = 7
geo_ntpp_B = 0.6

"""
   2) Chengdu 数据集范围
"""
cd_lat_min = 30.57
cd_lat_max = 30.72
cd_lng_min = 103.98
cd_lng_max = 104.14

cd_path0 = Total_path + "/run/RUN_Chengdu/Chengdu索引网格.xls"
cd_path1 = Total_path + "/run/RUN_Chengdu/Chengdu第一层网格.xls"
cd_path2 = Total_path + "/run/RUN_Chengdu/Chengdu第二层网格.xls"

cd_path3_1 = Total_path + "/run/RUN_Chengdu/Chengdu第三层网格01.xls"
cd_path3_2 = Total_path + "/run/RUN_Chengdu/Chengdu第三层网格02.xls"
cd_path3_3 = Total_path + "/run/RUN_Chengdu/Chengdu第三层网格03.xls"
cd_path3 = [cd_path3_1, cd_path3_2, cd_path3_3]

cd_thred_num01 = 5
cd_max_area = 22000
cd_thred_num02 = 0
cd_thred_midu = 5e-6

cd_d_thred = 400

cd_ntpp_rate = 6.3
cd_ntpp_B = 0.6

"""
   3) Shanghai 数据集范围
"""
sh_lat_min = 31.125
sh_lat_max = 31.35
sh_lng_min = 121.35
sh_lng_max = 121.57

sh_path0 = Total_path + "/run/RUN_Shanghai/Shanghai索引网格.xls"
sh_path1 = Total_path + "/run/RUN_Shanghai/Shanghai第一层网格.xls"
sh_path2 = Total_path + "/run/RUN_Shanghai/Shanghai第二层网格.xls"

sh_path3_1 = Total_path + "/run/RUN_Shanghai/Shanghai第三层网格01.xls"
sh_path3_2 = Total_path + "/run/RUN_Shanghai/Shanghai第三层网格02.xls"
sh_path3_3 = Total_path + "/run/RUN_Shanghai/Shanghai第三层网格03.xls"
sh_path3_4 = Total_path + "/run/RUN_Shanghai/Shanghai第三层网格04.xls"
sh_path3_5 = Total_path + "/run/RUN_Shanghai/Shanghai第三层网格05.xls"
sh_path3_6 = Total_path + "/run/RUN_Shanghai/Shanghai第三层网格06.xls"
sh_path3 = [sh_path3_1, sh_path3_2, sh_path3_3, sh_path3_4, sh_path3_5, sh_path3_6]

sh_thred_num01 = 5
sh_max_area = 22000
sh_thred_num02 = 0
sh_thred_midu = 6e-6

sh_d_thred = 800

sh_ntpp_rate = 6.4
sh_ntpp_B = 0.6


# POI兴趣点
POIs = {
    "银行": 92,
    "酒店": 85,
    "餐厅": 95,
    "商场": 90,
    "公园": 80,
    "博物馆": 70,
    "电影院": 75,
    "健身房": 80,
    "超市": 90,
    "咖啡馆": 80,
    "快餐店": 85,
    "书店": 70,
    "药店": 85,
    "面包店": 80,
    "酒吧": 50,
    "美发店": 75,
    "美容院": 70,
    "洗衣店": 65,
    "体育馆": 40,
    "游乐场": 75,
    "动物园": 70,
    "夜市": 30,
    "社区中心": 65,
    "游泳池": 60,
    "健身俱乐部": 70,
    "果蔬市场": 75,
    "家电店": 65,
    "家居店": 65,
    "电动车租赁": 55,
    "旅游景点": 80,
    "露营地": 25,
    "汽车站": 85,
    "火车站": 90,
    "公交站": 85,
    "加油站": 90,
    "摄影店": 55,
    "电脑店": 60,
    "电子商店": 65,
    "珠宝店": 40,
    "运动器材店": 55,
    "冰淇淋店": 60,
    "小吃摊": 75,
    "茶馆": 65,
    "酒类专卖店": 60,
    "沙龙": 65,
    "音乐厅": 70,
    "展览中心": 65,
    "艺术画廊": 60,
    "文化中心": 65,
    "健康食品店": 55,
    "宠物店": 70,
    "游艇俱乐部": 60,
    "滑雪场": 35,
    "高尔夫球场": 60,
    "水上乐园": 65,
    "电玩厅": 65,
    "DIY工作室": 55,
    "手工艺品店": 55,
    "车站候车室": 75,
    "美术馆": 70,
    "剧院": 75,
    "保健中心": 65,
    "调酒学校": 5,
    "珠宝设计店": 50,
    "陶艺工作室": 55,
    "瑜伽馆": 65,
    "养生馆": 30,
    "飞行俱乐部": 50,
    "植物园": 70,
    "博物馆商店": 60,
    "二手书店": 65,
    "特色餐饮店": 70,
    "酒庄": 60,
    "水果店": 85,
    "彩妆店": 65,
    "烘焙坊": 70,
    "美容学校": 50,
    "戏水乐园": 65,
    "水族馆": 70,
    "模拟游戏店": 15,
    "马术俱乐部": 50,
    "定制礼品店": 55,
    "外卖店": 88,
    "社区花园": 65,
    "摄影展": 35,
    "音乐商店": 55,
    "健身舞教室": 60,
    "马拉松赛事": 10,
    "艺术工作坊": 55,
    "黑暗料理店": 20,
    "科普教育中心": 60,
    "秋游场所": 65,
    "摄影基地": 55,
    "船屋": 5,
    "手表店": 45,
    "汽车维修店": 60,
    "烧烤店": 70,
    "夜总会": 10,
    "露天市场": 75,
    "文化节": 65,
    "古玩店": 55,
    "星空露营": 60,
    "手表维修店": 55,
    "运动会": 5,
    "周末市场": 75
}


