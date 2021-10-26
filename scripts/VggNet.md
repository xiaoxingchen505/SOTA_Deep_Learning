# VggNet

## 相关工作

1. AlexNet：借鉴卷积模型结构
2. ZFNet： 借鉴其采用小卷积核思想
3. OverFeat：借鉴全卷积，实现高效的稠密（Dense）预测
4. NIN：尝试1*1卷积

## 研究意义

1. 开启小卷积核时代：3*3卷积核成为主流模型
2. 作为各类图像任务的骨干网络结构：分类、定位、检测、分割一系列图像任务大都有VGG为骨干
网络的尝试


## Abstract

1. 本文主题：在大规模图像识别任务中，探究卷积网络深度对分类准确率的影响
2. 主要工作：研究3*3卷积核增加网络模型深度的卷积网络的识别性能，同时将模型加深到16-19层
3. 本文成绩：VGG在ILSVRC-2014获得了定位任务冠军和分类任务亚军
4. 泛化能力：VGG不仅在ILSVRC获得好成绩，在别的数据集中表现依旧优异
5. 开源贡献：开源两个最优模型，以加速计算机视觉中深度特征表示的进一步研究


## 模型结构

### 和AlexNet的相同点：
1. 5个maxpool 
2. maxpool后，特征图通道数翻倍直至512 
3. 3个FC层进行分类输出 
4. maxpool之间采用多个卷积层堆叠，对特征进行提取和抽样

### 演变过程特点
A：11层卷积, 其中maxpooling为2*2，且stride为2，意味着每一个池化层后，特征图的尺寸减半。

A-LRN：基于A增加一个LRN

B： 第1，2个block中增加1个卷积3*3卷积

C： 第3， 4， 5个block分别增加1个1*1卷积，表明增加非线性有益于指标提升

D：第3， 4， 5个block的1*1卷积替换为3*3，

E：第3， 4， 5个block再分别增加1个3*3卷积

为什么从11层开始？GoodFellow在2014年有一个applied deep ConvNets(11 weight layers) to the task of street number recognition

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg1.png">

### Vgg结构

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg2.png">

右边图中间少了一个conv3-256

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg3.png">

### Vgg结构特点

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg4.png">

1. 堆叠使用 3 x 3 卷积核， 2个3 x 3 卷积核等价于 1个5 x 5卷积核，3个3x3卷积核等价于1个7x7卷积核

    * 增加非线性激活函数，增加特征抽象能力
    * 减少训练参数
    * 可以看成7x7卷积核的正则化，强迫7x7分解为3x3

假设输入，输出通道均为C个通道：

一个7x7卷积核所需的参数量：7*7*C*C = 49 C^2

三个3x3卷积核所需的参数量：3*(3*3*C*C) = 27 C^2

参数减少比：(49-27)/49 ≈ 44%

2. 尝试1x1卷积，借鉴Network in Network(NIN)，引入利用1x1卷积，增加非线性激活函数，提升模型效果。


## 训练技巧 & 测试技巧

### 数据增强和尺度扰动(Scale Jittering)

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg9.png">

### 预训练模型初始化

深度神经网络对初始化敏感
1. 深度加深时，用浅层网络初始化, B，C，D，E用A模型初始化
2. Multi-scale训练时，用小尺度初始化, S=384时，用S=256模型初始化, S=[256, 512]时，用S=384模型初始化

### 多尺度测试

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg10.png">


### 稠密测试：

稠密测试（Dense test） ：将FC层转换为卷积操作，变为全卷积网络，实现任意尺寸图片输入

1. 经过全卷积网络得到 N* N* 1000 特征图
2. 在通道维度上求和（sum pool）计算平均值，得到1*1000 输出向

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg11.png" width="600" height="300">

### Multi-crop 测试

借鉴AlexNet与GoogLeNet，对图片进行Multi-crop，裁剪大小为224* 224，并水平翻转1张图，缩放至3种尺寸，然后每种尺寸裁剪出50张图片；50 = 5*5*2

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg12.png" width="600" height="300">

## 实验结果及分析

### Single scale evaluation

S为固定值时：Q = S， S为随机值时：Q = 0.5（S_min + S_max)

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg5.png">

结论：
1. 误差随深度加深而降低，当模型到达19层时，误差饱和，不再下降
2. 增加1*1有助于性能提升
3. 训练时加入尺度扰动，有助于性能提升
4. B模型中，3*3替换为5*5卷积，top1下降7%

### Multi scale evaluation
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg6.png">

方法1 :
Q = [S-32, S, S+32]

方法2：
Q = (S_min, 0.5*(S_min + S_max), S_max)

1. 测试时采用Scale jittering 有助于性能提升。

### Multi crop evaluation

方法:
等步长的滑动224* 224的窗口进行裁剪，在尺度为Q的图像上裁剪5*5=25张图片，然后再进行水平翻转，得到50张图片，结合三个Q值，一张图片得到150张图片输入到模型中

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg7.png">

结 论 ：
1. multi-crop优于dense
2. multi-crop结合dense，可形成互补，达到最优结


### Convnet fusion

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vgg8.png">

方 法 : ILSVRC中 ，多模型融合已经是常规操作。ILSVRC中提交的模型为7个模型融合采用最优的两个模型结合multi-crop和dense
* D / [ 2 5 6 , 5 1 2 ] / 2 5 6 , 3 8 4 , 5 1 2
* E / [ 2 5 6 , 5 1 2 ] / 2 5 6 , 3 8 4 , 5 1 2

## Vgg代码


[代码](code/VGG-代码/B_VGG)