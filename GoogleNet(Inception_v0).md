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