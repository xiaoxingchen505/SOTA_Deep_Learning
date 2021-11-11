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
.conda\envs\pytorch_1.4_gpu\Lib\site-packages\torchvision\models\resnet.py

## 预热训练 Warmup
避免一开始较大学习率导致模型的不稳定，因而一开始训练时用较小的学习率训练一个epochs，然后恢复正常学习率

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res6.png">

## 实验结果与分析
实验1：验证residual learning可解决网络退化问题，可训练更深网
络
ILSVRC top-1 error:
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res8.png">
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/res7.png">

## ResNet代码


[代码](code/VGG-代码/B_VGG)