# ResNet
Author: Kaiming He， Xiangyu Zhang ，Shaoqing Ren， Jian Sun (MSRA)



## 相关工作

1. Highway Network：首个成功训练成百上千层（100层及900层）的卷积神经网络 思路：借鉴LSTM，引入门控单元，将传统前向传播增加一条计算路
径.


## 研究意义

1. 简洁高效的ResNet受到工业界宠爱，自提出以来已经成为工业界最受欢迎的卷积神经网络结构
2. 近代卷积神经网络发展史的又一里程碑，突破千层网络，跳层连接成为标配



## 残差结构 Architecture Of Residual Learning

Residual learning：让网络层拟合H(x)-x， 而非H(x)

注：整个building block仍旧拟合H(x) ，注意区分building block与网络层的差异，两者不一定等价
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res1.png">



* 问：为什么拟合F（x）?
  答：提供building block更容易学到恒等映射（identity mapping）的可能

* 问：为什么拟合F（x）就使得building block容易学到恒等映射？
  答：如果网络层的输出为0的话，它就是一个恒等映射。比如：H(x) = F(x) + x，如果F(x)= 0，那么H(x) = x，那么这就是一个identity mapping

* 问：为什么要恒等映射？
  答：让深层网络不至于比浅层网络差

* 问：为什么深层网络比浅层网络差？
  答：网络退化问题

### 网络退化（degradation problem)
越深的网络拟合能力越强，因此越深的网络训练误差应该越低，但实际相反

原因：并非过拟合，而是网络优化困难

假设我们有一个18层的网络，我们加入building block，使其成为34层，若building block的网络层能学习到恒等映射，34层网络至少能与18层网络有相同性能

问：如何让额外的网络层更容易的学习到恒等映射？
答：skip connection == residual learning == shortcut connection

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res2.png">


### Shortcut Mapping

Identity 与 F(x)结合形式讨论：

1. A-全零填充：维度增加部分采用零来填充
2. B-网络层映射：当维度发生变化时，通过网络层映射 (例如：1*1 卷积) 特征图至相同纬度
3. C-所有Shortcut均通过网络层映射（例如：1*1卷积）

### Shortcut Mapping有利于反向传播
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res3.png">


## 模型结构

划分为6个stage

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res4.png">

1. 头部迅速降低分辨率
2. 4阶段残差结构堆叠
3. 池化+FC层输出
4. 从CONV3开始，每一个CONV的第一个building block的stride为2，这样输入的尺度为下降为一半，往后以此类推。


### 残差结构堆叠两种方式

Basic：两个3*3卷积堆叠
Bottleneck：利用1*1卷积减少计算量

Bottleneck：
第一个1*1下降1/4通道数
第二个1*1提升4倍通道数
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res5.png">

bottleneck中的4在resnet中是一个超参数：
```python
class Bottleneck(nn.Module):
    expansion = 4
```
位于.conda\envs\pytorch_1.4_gpu\Lib\site-packages\torchvision\models\resnet.py

## 预热训练 Warmup
避免一开始较大学习率导致模型的不稳定，因而一开始训练时用较小的学习率训练一个epochs，然后恢复正常学习率

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res6.png">

## 实验结果与分析

### 实验1：验证residual learning可解决网络退化问题，可训练更深网
络
ILSVRC top-1 error:
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res8.png">
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res7.png">

### 实验2：横纵对比，shortcut策略（ABC）及层数

1. A-全零填充：维度增加的部分采用零来填充
2. B-网络层映射：当维度发生变化时，通过网络层映射特征图至相同维度
3. C-所有Shortcut均采用网络层映射

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res9.png">



### 实验3：成功训练千层神经网络

Cifar-10数据集上成功训练1202层卷积网络
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res10.png">

在这个图中，1202层的网络不如110层网络是因为过拟合现象。

### 实验4：残差学习输出神经元尺度

统计每个卷积+BN层输出的神经元尺度大小，以标准差来衡量尺度

结论：ResNet输出比plain小，表明带残差学习的结构比不带残差学习时，输出更偏向0，从而更近似于恒等映射

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res11.png">


##  论文总结

### 关键点&创新点

* 引入shortcut connection，让网络信息有效传播，梯度反传顺畅，使得数千层卷积神经网络都可以收敛
注：本文中：shortcut connection == skip connection == identity mapping


### 启发点

1. 大部分的梯度消失与爆炸问题，可通过良好初始化或者中间层的标准化来解决。

An obstacle to answering this question was the notorious problem of vanishing/exploding gradients [1, 9],
which hamper convergence from the beginning. This problem, however, has been largely addressed by
normalized initialization [23, 9, 37, 13] and intermediate normalization layers （1 Introduction p2）

2. shortcut connection有很多种方式，本文主要用的是恒等映射，即什么也不操作的往后传播

In our case, the shortcut connections simply perform identity mapping. (1 Introduction p6)

3. highway network的shortcut connection依赖参数控制，resnet不需要

These gates are data-dependent and have parameters, in contrast to our identity shortcuts that are
parameter-free.(2 Related Work p4)

4. 恒等映射形式的shortcut connection是从网络退化问题中思考而来

This reformulation ( H(x ) = F(x) + x )is motivated by the counterintuitive phenomena about the
degradation problem.(3.1 Residual learning)

5. 借鉴VGG，本文模型设计原则：1.处理相同大小特征图，卷积核数量一样；2.特征图分辨率降低时，通道数翻倍

two simple design rules: (i) for the same output feature map size, the layers have the same number of
filters; and (ii) if the feature map size is halved, the number of filters is doubled so as to preserve
the time complexity per layer. （3.3 Network Architectures p2）

6. 当特征图分辨率变化时，shortcut connection同样采用stride=2进行处理

For both options, when the shortcuts go across feature maps of two sizes, they are performed with a
stride of 2. （3.3 Network Architectures p4）

7. bottleneck 中两个1*1卷积分别用于减少通道数和增加/保存通道数

The three layers are 1×1, 3×3, and 1×1 convolutions, where the 1×1 layers are responsible for reducing and
then increasing (restoring). (4.1 Imagenet Classification Deeper Bottleneck


8. 模型集成采用6种不同深度的ResNet结构，可以借鉴其思路
We combine six models of different depth to form an ensemble (only with two 152-layer ones at the time of
submitting) (4.1 Imagenet Classification Comparisons with State-of-the-art Methods.)


9. cifar-10数据集上的ResNet-110, 第一个epochs采用较小学习率，来加速模型收敛
We further explore n = 18 that leads to a 110-layer ResNet. In this case, we find that the initial learning rate
of 0.1 is slightly too large to start converging5. So we use 0.01 to warm up the training until the training
error is below 80% (about 400 iterations), and then go back to 0.1 and continue training. (4.2. CIFAR-10 and Analysis p6)

10. cifar-10数据集上，ResNet-1202比110要差，原因可能是过拟合，而不是网络退化
But there are still open problems on such aggressively deep models. The testing result of this 1202-layer
network is worse than that of our 110-layer network. We argue that this is because of overfitting. (4.2.
Exploring Over 1000 layers)

11. 本文无conclusion，没有对未来可研究的内容进行总结，in the future 贯穿在文中有两处
  * 导致网络退化的训练困难问题目前还未清楚，需要在将来研究
The reason for such optimization difficulties will be studied in the future.
  * 本文重点在于研究shortcut connection，所以ResNet未用maxout or dropout之类的正则化方法，将来可考虑加入这些正
则化方法，进一步提升模型性能
But combining with stronger regularization may improve results, which we will study in the future

12. 模型的思考
为什么是14*14的时候改动building block数量？
inception-v3中的非对称卷积分解建议在12-20之间，从两种策略中是否总结出某种经验结论？

13. “无辜”的VGG，被各大模型用于进行参数量的对比
自GoogLeNet之后，输出几乎不采用多个全连接层堆叠形式，大大减少了参数量

14. Residual learning 提供网络特殊的结构，使也有机会学习到identity mapping，但具体网络学不学，交给模型自己
回顾GoogLeNet-V2的Batch Normalization中的γ和β，提供这个线性变换，让模型有机会恢复标准化前的尺度，但恢复还是不恢复，
交给模型自己
人类完成的是逻辑上的改变，给模型更多的可能性，然后论文中可以说，一切交给模型自己决定


## ResNet代码


[代码](code/ResNet-code)