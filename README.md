# **TDSF: Trajectory-preserving Method of Dual-Strategy Fusion with** **Differential Privacy in LBS**

 ![System model.jpg](https://github.com/hxlabc/TDSF/blob/main/pic/System%20model.jpg?raw=true)

​												**图1：系统模型**

**第一作者：**XianLiang He



## 摘要

当公众使用基于位置的服务(LBS)时，产生了大量的轨迹数据,他们的位置信息会不断地暴露。然而，在没有额外保护的情况下将轨迹提供给LBS可能造成轨迹的位置隐私和相关性隐私的泄露。目前的大多数方法仅通过调整隐私预算的分配来保护轨迹的位置隐私，没有采用多种策略联合保护位置隐私和相关性隐私。此外，这些方法也很难平衡轨迹的数据可用性和隐私性。为了解决上述挑战，我们提出了一种双策略融合保护的差分隐私轨迹保护机制(TDSF)。该机制位于轨迹处理中心服务器(TPC)中。它的目标是通过双策略分别保护轨迹内的位置和相关性隐私，最大程度预防位置隐私泄露和轨迹内推理攻击，并保证较高的数据可用性。

 

## 研究工作

我们的方案主要包括五个模块：密度网格的划分，转移相关性矩阵的计算，轨迹中关键位置（高相关性位置）的提取，隐私预算合理分配以及差分隐私保护算法。由于TPC中也存储了人口分布信息，因此可以根据人口密度对一定范围内的地理区域进行多层自适应网格划分。然后统计大量用户运动轨迹在网格上的转移情况训练出一个转移相关性矩阵。这个矩阵也可以反映该地理区域上人们的轨迹运动情况。由于人口分布和轨迹数据库都是不断动态更新的，因此划分的网格以及转移相关性矩阵里的数值也是随着时间动态更新的。接着通过该矩阵对用户上传的轨迹进行高相关性点的提取，我们对它们采用新颖的基于指数机制的相关性差分隐私策略。非高相关性位置则采用传统的Geo-Indistinguishability加噪策略，由于它们不泄露太多隐私，所以适当减少噪声，从而保证数据的高可用性。在这两种策略中，前者专注于保护相关性隐私，后者专注于保护位置隐私。将两者结合能够确保轨迹中这两种类型的位置点都能得到既精确又有效的防护。再整个加噪过程中，我们也对轨迹中每个位置点自适应地分配隐私预算。最终加噪保护后的信息安全地释放给LBS，用户再从LBS得到想要的查询结果。

<img src="https://github.com/hxlabc/TDSF/blob/main/pic/Geospatial%20discretisation.jpg?raw=true" alt="Geospatial discretisation.jpg" style="zoom: 33%;" />

​									**图2：自适应三层网格划分**



<img src="https://github.com/hxlabc/TDSF/blob/main/pic/Calculation%20of%20transfer%20matrix.jpg?raw=true" alt="Calculation of transfer matrix.jpg" style="zoom:80%;" />

​									**图3：转移相关性矩阵**



<img src="https://github.com/hxlabc/TDSF/blob/main/pic/Correlation%20Perturbation%20Mechanism.jpg?raw=true" alt="Correlation Perturbation Mechanism.jpg" style="zoom: 50%;" />

​										    **图4：轨迹加噪**

## 研究成果

严格安全性分析表明，我们提出的TDSF机制可以很好地保护轨迹的位置和相关性隐私。在真实数据集上的实验表明，TDSF能够兼顾整条轨迹的高可用性和关键位置的强隐私性。这是因为随着隐私预算增加，轨迹的数据可用性增加导致非关键位置的隐私保护下降，TDSF为了弥补这一点，关键位置的相关隐私保护程度会随隐私预算增加而上升。这样使得数据可用性和隐私保护性达到了一定的平衡。因此，用户可以采用不同的隐私预算达到不同的目的，从而满足他们的个性化需求。

 

## 投稿期刊

**Computers & Security**  **CCF-B**类期刊（状态：Under Review）