# AlexNet

## 研究意义

1. 拉开卷积神经网络统治计算机视觉的序幕
2. 加速计算机视觉应用落地


## 模型特点

1. 5个卷积层和3个全连接层构成，共计6000万参数，65万个神经元。
2. 加快训练，采用非饱和激活函数，ReLu，使用GPU训练
3. 为减轻过拟合，采用Dropout

## 使用了 LRN

1、其中LRN就是局部响应归一化：

这个技术主要是深度学习训练时的一种提高准确度的技术方法。其中caffe、tensorflow等里面是很常见的方法，其跟激活函数是有区别的，LRN一般是在激活、池化后进行的一中处理方法。在CNN中使用重叠的最大池化。此前CNN中普遍使用平均池化，AlexNet全部使用最大池化，避免平均池化的模糊化效果。并且AlexNet中提出让步长比池化核的尺寸小，这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性。AlexNet提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

Hinton在2012年的Alexnet网络中给出其具体的计算公式如下：
![image](https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/alex1.png)

公式看上去比较复杂，但理解起来非常简单。i表示第i个核在位置（x,y）运用激活函数ReLU后的输出，n是同一位置上临近的kernal map的数目，N是kernal的总数。参数K,n,alpha，belta都是超参数，一般设置k=2,n=5,aloha=1*e-4,beta=0.75。

## AlexNet结构：


```java

a = 0
b = 1
c = 2

for i in range(c):
    print(i)

```