# 论文: ArcFace Additive Angular Margin Loss for Deep Face Recognition (CVPR2019)

## 一. 内容介绍
使用深度学习进行人脸识别的一个难点是，如何提高人脸特征的类间距离，如何减小类内距离。传统的softmax的方法，只能进行类别的区分，并不能达到前面的要求。所以，论文作者提出一个添加角度的惩罚的方法，来实现这一目的，达到了不错的效果。

## 二. 内容说明
1. 论文的贡献
论文主要有一下三个方面的特点:

有趣的设计， 论文将loss函数，转换到了，超球面，使loss函数有了一个很直观的解释。
有效果, 论文最后的结果达到了sota的效果。
简单, 只是简单在角度上添加了一个惩罚角度，而不用修改既有的网络架构，实现比较简单。
有效率，因为只是添加了一个角度的惩罚，使train后者prediction的执行速度，没有太大 的影响
2. 网络结构设计
从如下softmax的公式可以看出。如果令 [公式] = 1 (L2范式)， 然后把 [公式] 的大小缩放到s.
![image](https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/eq1.jpg)

根据向量夹角公式，这上式，就可以变形为:
![image](https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/eq2.png)

这样，就把softmax的loss，映射到了半径为s的超球面， [公式]对[公式]的角度。(这里X为backbone的输出特征向量)。 再在此角度上加上m度的惩罚，如下式:
![image](https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/eq3.png)

就完成了softmax的改造，

只需把softmax进行上面几步的改造就完成了。提高类间距离，减小 类内距离的效果。

3. 与softmax, shpereface, cosface的比较。
shpereface是在角度上乘以一个参数，达到提高类间距离，减小类内距离的效果。而cosface是在cosine的基础上在加上一个cosine的参数作为惩罚，达到提高类间距离，减小类内距离的效果。

![image](https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/eq4.jpg)

从图上看出， shpereface，和cosface以及论文的arcface都是类间的距离拉开了。而且， cosface和arcface的效果都不错。 接下来作者就提供了各种数据的效果实验，结论是arcface的效果基本上达到了sota水平。

## 三. 启发及评价
论文最大的亮点还是，把softmax转换到了超球面空间，试loss值有了一个很 明确的几何解释，这个是很有意思的一个观点。softmax作为传统的loss值求解的一部分，有 可能过于简单了。如果能映射到，其他空间，应该也有比较新颖的效果。

因为论文，是使用在人类分类上，只要求求出特征值，后面的arc-cosine的loss值在使用时可以直接丢掉，但是如果是直接分类的话，在预测时，就不太好处理了，是 直接把cosine的惩罚设置成0，就可以了吗？ 不知道效果如何。

## 四. 其他
涉及到的知识点:
facenet的triplet loss, Centre loss, Shpereface, Cosface,


转载于：https://zhuanlan.zhihu.com/p/112574720