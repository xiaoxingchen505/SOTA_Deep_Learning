



# Transformer: Attention Is All You Need


## 研究背景

在 ConvS2S 和 ByteNet 这两个模型中，学习两个距离很远的信号之间的关系所需要的计算运行次数是会线性增加的，这使得这两个模型很难去学习两个距离位置信号之间的关系。Tranformer 把这种计算运行次数减小到了常量 （constant），也就是说不会发生变化。作者提出的 Transformer架构，不同于 RNN 和 CNN 网络架构。 Transformer 完全建立于 self-attention 自注意力机制上。该模型除了在两个machine translation 任务上除了除了表现得非常好，同时可以允许并行计算，减少了大量的训练时间。


## 模型架构

Encoder-decoder 架构：

Encoder 一般是映射一组输入 (x1, ..., xn) 到一组连续的表示 z = (z1, ..., zn)。通过输入 z 到 decoder 中， decoder可以产生一组输出序列 (y1, ..., ym)。每一步都是 auto-regressive，也就是说每一步产生的输出都会在下一步作为额外的输入和新的输入一起产生下一个新的输出。

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans1.png">