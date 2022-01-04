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

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit4.png" width="800" height="400">

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/vit5.png" width="800" height="400">

