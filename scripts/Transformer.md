



# Transformer: Attention Is All You Need


## 研究背景

在 ConvS2S 和 ByteNet 这两个模型中，学习两个距离很远的信号之间的关系所需要的计算运行次数是会线性增加的，这使得这两个模型很难去学习两个距离位置很远的信号之间的关系。Tranformer 把这种计算运行次数减小到了常量 （constant），也就是说不会发生变化。作者提出的 Transformer架构，不同于 RNN 和 CNN 网络架构。 Transformer 完全建立于 self-attention 自注意力机制上。该模型除了在两个machine translation 任务上除了除了表现得非常好，同时可以允许并行计算，减少了大量的训练时间。其次，卷积神经网络可以同时关注到多个输出通道，一个输出通道可以去识别不一样的模式，作者提出了 multi-head attention 来模拟卷积神经网络多输出通道的一个效果。


## 模型架构

Encoder-decoder 架构：

Encoder 一般是映射一组输入 (x1, ..., xn) 到一组连续的表示 z = (z1, ..., zn) 向量。通过输入 z 到 decoder 中， decoder可以产生一组输出序列 (y1, ..., ym)。每一步都是 auto-regressive，也就是说每一步产生的输出都会在下一步作为额外的输入和新的输入一起产生下一个新的输出。

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans1.png">


### Encoder 和 Decoder 堆叠

* Encoder： Encoder 由 N = 6 个相同层组成。 每一层有两个次级层。 第一个是多头注意力机制， 第二个是 position-wise （input/output维度一样）的 feed-forward 网络。从上图能看到每次级层后都使用了残差链接(residual connection), 紧随着是一个normalization层。 

  每一个次基层的输出可以表示为：LayerNorm(x + Sublayer(x))， 其中 Sublayer(x) 是次基层其本身。为了实现残差链接，每个次级层输出的维度为 d_model = 512， 与 embedding 层的维度保持一致。

* Decoder：Decoder 也同样由 N = 6 个相同层组成。 除了与 encoder 一样每一层拥有两个相同的次级层，decoder 还插入了第三个次层级，这一层对 output embedding 进行了一次 multi-head attention 计算。Decoder 中的 self-attention 次级层也加入了一个masking，以此来保证每一次基于 position i 的预测只取决于小于 i 位置的已知 outputs。这样做的目的是为了保证训练过程中，每一个位置的预测只基于当前和之前的位置的信息，如果模型能够看到所有位置的信息，那这样的训练是无效的，因为 i 位置之后的信息包含了我们想要预测的答案。


## attention 机制：

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans2.png">