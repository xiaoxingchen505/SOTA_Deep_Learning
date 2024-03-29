



# Transformer: Attention Is All You Need

本篇的大部分笔记来自于李沐老师的视频，附上传送门：https://www.bilibili.com/video/BV1pu411o7BE

## 研究背景 :

在 ConvS2S 和 ByteNet 这两个模型中，学习两个距离很远的信号之间的关系所需要的计算运行次数是会线性增加的，这使得这两个模型很难去学习两个距离位置很远的信号之间的关系。Tranformer 把这种计算运行次数减小到了常量 （constant），也就是说不会发生变化。作者提出的 Transformer架构，不同于 RNN 和 CNN 网络架构。 Transformer 完全建立于 self-attention 自注意力机制上。该模型除了在两个machine translation 任务上除了除了表现得非常好，同时可以允许并行计算，减少了大量的训练时间。其次，卷积神经网络可以同时关注到多个输出通道，一个输出通道可以去识别不一样的模式，作者提出了 multi-head attention 来模拟卷积神经网络多输出通道的一个效果。


## 模型架构 :

Encoder-decoder 架构：

Encoder 一般是映射一组输入 (x1, ..., xn) 到一组连续的表示 z = (z1, ..., zn) 向量。通过输入 z 到 decoder 中， decoder可以产生一组输出序列 (y1, ..., ym)。每一步都是 auto-regressive，也就是说每一步产生的输出都会在下一步作为额外的输入和新的输入一起产生下一个新的输出。

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans1.png">


### Encoder 和 Decoder 堆叠 :

* Encoder： Encoder 由 N = 6 个相同层组成。 每一层有两个次级层。 第一个是多头注意力机制， 第二个是 position-wise （input/output维度一样）的 feed-forward 网络。从上图能看到每次级层后都使用了残差链接(residual connection), 紧随着是一个normalization层。 

  每一个次基层的输出可以表示为：LayerNorm(x + Sublayer(x))， 其中 Sublayer(x) 是次基层其本身。为了实现残差链接，每个次级层输出的维度为 d_model = 512， 与 embedding 层的维度保持一致。

* Decoder：Decoder 也同样由 N = 6 个相同层组成。 除了与 encoder 一样每一层拥有两个相同的次级层，decoder 还插入了第三个次层级，这一层对 output embedding 进行了一次 multi-head attention 计算。Decoder 中的 self-attention 次级层也加入了一个masking，以此来保证每一次基于 position i 的预测只取决于小于 i 位置的已知 outputs。这样做的目的是为了保证训练过程中，每一个位置的预测只基于当前和之前的位置的信息 (key-value pair)，如果模型能够看到所有位置的信息，那这样的训练是作弊的，因为 i 位置之后的信息包含了我们想要预测的答案。Mask 的实际操作是在每一次训练时，我们把 position > i 的值替换成一个比较大的负数，这样在之后进入 softmax 层的时候就会变成 0。在计算权重的时候，i 位置信息之后所对应的权重都会变成 0.


## attention 机制：

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans2.png">

Attention 机制可以理解为将一个 query 和一些 key-value pairs 映射成一个输出的函数。其中 query， keys， values，output 都是向量。Output是 value 的 加权和。每一个value的权重是这个 value 对应的 key 和查询的 query 的相似度来计算的。


### Scaled Dot-Product Attention :

作者在 Transformer 模型中提出了 'Scaled Dot-Product Attention' 机制。 其中 query 和 key 的维度都是 d_k, value 的 维度是 d_v。通过对每一个 query 和 key 做内积来得到相似度。两个向量的内积值越大，代表这两个向量的相似度越高。然后把这个值除以<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans3.png"> （向量的长度），然后通过一个 softmax 函数就可以得到权重。假如给定一个 query， 和 n 个 key-value pair 的话，那么就会算出 n 个值，因为这个 query 会跟每个 key 做内积。算出来以后再放进 softmax 函数就会得到n 个非负的而且加起来和等于一的一个权重，然后我们把这些权重作用在我们的 value 上面， 就可以得到我们的输出了。

在实际问题中，我们需要用矩阵的形式来进行计算, 公式如下：

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans4.png">

举个例子：

假设我们的 query 有 n 行， 列数为 d_k, 我们的 keys 有 m 行，列数也为 d_k， 这样我们用内积可以得到一个 n 乘以 m 的矩阵， 这个矩阵中，每一行是这个 query 对所有 keys 的内积值，然后再除以 <img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans3.png">， 再进入 softmax 函数，其实就是对每一行做 softmax， 行与行之间是独立的，这样就可以得到我们需要的权重。然后再乘以我们的 V ，V 是一个有 m 行，列数为 d_v 的矩阵。 把我们的权重矩阵和 V 做内积就可以得到一个 n 乘以 d_v 的矩阵，也就是我们的输出。

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans5.png">

### 为什么要除以 d_k ? ：

当我们的值比较大的时候，最大值和最小值的差距会非常大，以至于在softmax层后，最大的一部分值会无限接近于 1，最小的一部分值会无限接近于 0， 我们的值就会更加向两端靠拢，在这种情况下算梯度的话，梯度会非常小。如果我们能够把我们的值除以 <img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans3.png">，把数值进行缩放，我们可以得到更好的梯度。

### Multi-Head Attention 多头注意力机制 ：

与其做一个单个的注意力函数，作者把整个的 query，key，value 分别投影到了一个低维度，投影 h 次。然后再做 h 次的注意力函数。每一个的函数的输出都会被并到一起，然后再投影回来得到我的最终输出。这里，linear层就是用来投影到比较低的维度。通过 h 次的 Scaled Dot-Product Attention 计算，我们会得到 h 个输出，然后我们把这些向量全部合并到一起，最后做一次线性的投影，就得到了我们的 multi-head attention。

### 为什么需要 multi-head attention？：

如果我们回过头来看 dot-product attention，你会发现我们没有什么可以学的参数，你的距离函数就是你的内积。但是有时候为了识别不一样的那些模式，作者希望模型可以有一些不一样的计算相似度的办法。在multi-head attention中，把 Q,K,V 投影到低维度，这个投影的 w 是可以学习的，也就是说，我给模型 h 次机会， 希望模型可以学到不一样的投影的方法，使得在那个投影进去的那个度量空间里面能够去匹配不同的模式，它们所需要的一些相似函数，最后再把这些东西拿回来做一次投影。所以，这一点有点像我们在卷积神经网络当中，我们有多个输出通道的感觉。

### Multi-Head Attention 公式 ：

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans6.png">

在这里我们可以看到，我们把多个 head 的输出给 concat 起来，再投影到一个 W_o 里面。对每一个 head，它就是把 Q,K,V 通过一个不同的，可以学习的 W_Q, W_K, W_V, 投影到一个低维上面，再做我们之前提到过的注意力函数。在实际模型中，作者使用了 h = 8， 也就是 8 个头， 在投影时，投影的维度就是输出的维度除以 h ， 然后我们再算注意力函数，然后再投影回来。在上面的公式中，我们可以看到有需要很多矩阵乘法，但是在实际算法中，我们可以用一次矩阵乘法就可以实现。


### 模型中 Multi-Head attention 的实际用法：

* Encoder的注意力层： 

    在 encoder 中，假设输入的句子的长度为 n，它的输出其实是一个 n 个长为 d 的向量。（假设我们的pn大小设成 1）。在这里，我们拿到这个向量，我们复制成了三份，也就是说同样一个东西，既作为 key，也作为 value，也作为 query，所以这样的机制就被叫做自注意力，也就是说我们的 key，value，query 其实就是一个东西，就是自己本身。在这里，我们知道了我们输入了 n 个 query，那么每一个 query 我会拿到一个输出，那么意味着我会有 n 个输出。而且这个输出和 value 的长度是一样的，那么我输出的那个维度其实也是 d。其实我们的输出就是我们 value 的一个加权和，权重是来自于 query 和 key 的一些东西。所以这个机制实际上本身就是你的输出的一个加权的一个和。

    <img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans7.png">

    上图我们可以看到，不考虑多头注意力机制的情况下，权重其实就是你自己本身跟别的输入的别的向量所计算的相似度，其权重跟自己的向量算肯定是最大的。在多头注意力机制中，我们说过有投影，所以这个地方我们会学习 h 个不一样的距离空间出来，使得计算出来的东西会有一点点不一样了。

* Decoder的第一个注意力层： 

    解码器其实也是一样的，只不过长度可能变成了一个长度为 m 的样子，然后你的维度其实也是一样的。唯一不一样的就是这里有个masked的机制，也就是在解码的时候，计算权重的时候，要把 query 所对应的位置往后的权重设为 0。

* Decoder的第二个注意力层：

    这里，输入进去的 key 和 value 来自于你的编码器的输出，然后你的 query 来自于你的decoder第一个注意力层的输出，这里就不是自注意力机制了。每一个 query，我们算一个输出，那么输出我们知道是来自于 value 的一个加权和，那这里也就是来自于编码器输出的加权和。这个 attention 层所做的事情，其实就是有效的把我们的编码器里面的一些输出，根据我想要的东西提取出来。


### Position-wise Feed-Forward Networks ：

除了注意力层，每一个在 encoder 和 decoder 中有一个全连接的多层感知机，也就是 MLP。不一样的是，这里作者是把一个MLP 对每一个词作用一次，然后对每个词作用的是同样一个 MLP.

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans8.png">

在这个公式中，我们的作者通过 W_1 把输入的维度 512 给投影到了 2048，然后又通过 W_2，投影回了 512。所以这个东西其实就是一个单隐藏层的 MLP，中间的隐藏层，把你的输入扩大四倍，最后输出的时候也会回到你的输入的大小。用 Pytorch来实现的话，就是把两个线性层放到一起，你都不需要改任何参数，因为 Pytorch中，当你的输入是一个 3D 的时候，他默认就是在最后一个维度做计算。


### Transformer 和 RNN 的区别 : 

RNN 是把上一个时刻的信息，输出传入下一个时候做输入。但是在Transformer里面，它是通过一个attention层，然后再全局的去拉到整个序列里的信息，然后再用MLP做语义的转换。


### Embedding 和 softmax ：

Embedding 其实就是对于任何一个词，我学习一个长为 d 的向量来表示它，d 在这个作者模型里面就是 d_model = 512 ， 你的编码器和解码器都需要一个 embedding。 作者也使用了一个线性层和 softmax 函数来把解码器的输出转化为下一个预测的 token 的概率。 在最后 softmax 前面的那个线性层中，也需要一个embedding。 这三个 embeddings 是一样的权重，这样训练起来会简单一点。在这些 embeddings 中，我们把权重乘以 <img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans9.png">。主要是为了控制 embedding的 scale，让 embeddings 和 positional encoding相加的时候，scale是差不多的。其实也就是把 embeddings 的结果变小一点，给后面加 positional encoding 腾出地方。

### Positional Encoding 位置编码 :

位置编码是为了让 attention 机制保留一些时序上的信息，做法是在输入上加入时序信息。假设我们有一个词在 embedding 是维度为 512 的向量，同样我们用一个长为 512 的向量来表示一个数字，来表示这个词在输入中的位置。具体这个值是用周期不一样的 sin 和 cos 函数的值算出来的。

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans10.png">

然后这个长为 512 的记录了时序信息的向量，跟我们的 embeddings 相加，就完成了把时序信息，加入到我们的输入里。之前我们把我们的embeddings 乘以了 <img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans9.png">，也是让我们的值在 -1 到 1 之间，因为我们的positional encoding 是通过 sin 和 cos 函数算出来的，所以数字也是差不多到 1 到 -1 之间。


## 为什么要用 self-attention

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/trans11.png">

作者用这个图表来解释了为什么 self-attention 更好，其实主要是也是说明 RNN 的序列化计算，导致在并行上是比较吃亏的。其次，最初点的信息到最终点的信息，需要走过 n 步，所以 RNN 经常被批评对长序列的信息处理得不好，而不像 attention 一样，直接一步就可以过去。这三个模型的复杂度是差不多的，但是在计算上，attention 和卷积会好一点，另外一个就是说 attention 在信息的糅合度上会好一点，所以你可以认为这个地方还是影响了一些东西。但是实际上，attention 对于整个模型的假设做了更少，导致 attention 需要更多的数据，更大的模型才能训练出来，达到跟 RNN 和 CNN 同样的效果。

