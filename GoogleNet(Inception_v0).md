# GoogleNet (Inception)

## 研究意义

1. 开启多尺度卷积时代
2. 拉开 1*1卷积广泛应用序幕
3. 为GoogleNet系列开辟道路


## 模型特点

1. Inception的特点是提高计算利用率，增加网络深度和宽度，参数少量增加
2. 借鉴Hebbian理论和多尺度处理

## Inception Module的特点

1. 多尺度
2. 1*1 卷积降维，信息融合
3. 3*3 max pooling 保留了特征图数量


3 x 3 pooling可让特征图通道数增加，且用较少计算量，缺点：数据量激增。解决方法：引入1*1卷积压缩厚度

![image](https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/inception1.png)

1乘1卷积的作用主要还是把深度（通道数）减小，也就是映射到更低的维度，同时保证输入的长宽尺寸不变。

## GoogLeNet结构：

1. 三阶段：Conv-pool-conv-pool 快速降低分辨率; 堆叠Inception; FC层分类输出
2. 堆叠使用Inception Module，达22层
3. 增加两个辅助损失，缓解梯度消失 (中间层特征具有分类能力)
4. 5个Block
5. 5次分辨率下降
6. 卷积核数量变化魔幻
7. 输出层为1层的FC层

![image](https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/inception2.png)

## 稀疏结构-特征图通道的分解

672个特征图分解为四个部分

1. 1x1卷积核提取的128个通道
2. 3x3卷积核提取的192个通道
3. 5x5卷积核提取的96个通道
4. 3x3池化提取的256个通道

打破均匀分布，相关性强的特征聚集在一起。

![image](https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/inception3.png)