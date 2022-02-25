# Vision Transformer
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scal



## 相关工作

Transformer缺少一些inductive biases，也就是归纳偏置。CNN的inductive bias应该是locality和spatial invariance, locality可以理解为划窗部分的操作是默认图片相邻的部分是相关的，spatial invariance又叫做translation equivalence. 即为 f(g(x)) = g(f(x))，假设 f 是卷积，g 是平移，无论哪个函数先执行，最后的结果是一样的。


## Transformer的优势

1. 并行计算
2. 全局视野
3. 灵活的堆叠能力


## ViT结构

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit2.png">

假设我们有一个输入 x 的维度是 224 x 224 x 3， 如果我们使用 16 x 16 的 patch size大小，我们可以获得 224^2 / 16^ 2 = 196 个 token， 每一个图像块的维度是 16 x 16 x 3 = 768， 3 是RGB通道数。所以我们就把原来的图像变成了有 196 个patch，每一个patch的维度是768，也就是 196 x 768。最后我们还需要加入一个位置编码信息，于是我们传入transformer encoder的embedded patches的维度就是 (196 + 1) x 768。在通过Norm层以后，还是 197 x 768。 然后我们要进去 Multi-head attention层，也就是 K Q V， 假设我们用的ViT的base版本，其中的multi head是取的12个head，其中K Q V每一个的维度都是 197 x (768/12) = 197 x 64，因为我们一共有12个头 (head)。最后经过拼接，出来的维度还是 197 x 768. MLP层会把输入放大四倍，也就是 197 x 3012， 然后再缩小回 197 x 768。

## 研究成果

* ViT和ResNet Baseline取得了不相上下的结果
* 展示了在计算机视觉中使用纯Transformer结构的可能


## Multi-Head attention


### self-attention 计算

Scaled dot product

公式：

实际上是在进行相似度计算，计算每个q和每个k的相似度

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit3.png">

为什么这个地方需要除以 dk 的平方根呢？ 

答：作者怀疑 dk 的值如果特别大的话，点积过后会产生比较大的值，会导致 softmax 函数进入到一个梯度非常小的值域。为了防止这个问题，所以点积除以了 dk的开平方根

为什么是 dk 的开平方根 ？

答：假设 q 和 k 满足均值为 0 ， 方差为 1 的标准正态分布， 那么他们的点积的均值和方差分别为 0 以及 dk

综上所述：这样做的好处是避免较大的极值，较大的数值会导致softmax之后值更极端，softmax的极端值会导致梯度消失。

### Q,K,V的获得

本质： input的线性变换

计算: 矩阵乘法

实现：nn.Linear


### self-attention结构
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit4.png" width="800" height="400">

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit5.png" width="800" height="400">

最后的sum z1是通过v1，v2，v3 以 elemental-wise 乘以对应的softmax值，然后相加。 公式：z1 = v1 * 0.8 + v2 * 0.15 + v3 * 0.05。通过这种方法，我们可以逐渐算出z2，z3，在transformer架构中，计算可以矩阵化 (并行)

sum之后的z的意义：一词多义，比如date 会有约会和日期两个意思，这里的输出就是将单词的意思 -> 句中意思的转化

### MultiHead Attention

有多个Wq, Wk, Wv上述操作重复多次，结果concat一起.

为什么使用 Multihead Attention? 答： 给注意力提供多种可能性

例如： Conditional DETR (用于目标检测) 发现不同的 head 会 focus 到物体的不同边。


### 输入端适配 Input Adaptation

如果只有原始输出的9个向量，用哪个向量来分类都不好，全用计算量又很大，所以加一个可学习的vector，也就是patch 0来整合信息。


## 位置编码 Positional Encoding

图像切分重排后失去了位置信息，并且Transformer的内部运算是空间信息无关的，所以需要把位置信息编码重新传进网络，ViT使用了一个可学习的vector来编码，编码vector和patch vector直接相加组成输入

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit6.png">





问：为什么直接相加，而不是concat ?


答：因为相加是concat的一种特例

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit7.png">


## VIT结构的数据流

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit7.png">



## 训练方法

### 大规模使用Pre-Train
先在大数据集上预训练，然后到小数据集上Fine Tune，迁移过去后，需要把原本的MLP Head换掉，换成对应类别数的FC层（和过去一样），处理不同尺寸输入的时候需要对Positional Encoding的结果进行插值。


### 关于positional encoding的插值

不同的input size和patch size会切出不同数量的patch，所以编号的方法需要缩放


## 结果分析

### Attention距离和网络层数的关系

Attention的距离可以等价为Conv中的感受野大小，可以看到越深的层数，Attention跨越的距离越远，但是在最底层，也有的head可以覆盖到很远的距离，这说明他们确实在负责Global信息整合

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit8.png">



## 论文总结

### 关键点

* 模型结构——Transformer Encoder
* 输入端适配——切分图片再重排
* 位置编码——可学习的vector来表示

### 创新点


* 纯Transformer做分类任务
* 简单的输入端适配即可使用
* 做了大量的实验揭示了纯Transformer做CV的可能性。