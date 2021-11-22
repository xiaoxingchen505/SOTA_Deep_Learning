

# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis


Keywords: scene representation, view synthesis, image-based rendering, volume rendering, 3D deep learning

## 相关方法 Related works：

recent work has investigated the implicit representation of continuous 3D shapes as level sets by optimizing deep networks that map xyz coordinates to signed distance functions or occupancy fields.

然而这些模型的限制是它们必须需要 ground truth 3D geometry， 通常是合成的3D shape数据集 比如 shapeNet这个方法. 随后的工作放宽了这一要求，仅使用2D 图片，通过制定可微分的渲染函数 (differential rendering functions) 来构建真实3D形状允许对神经隐式形状表示 (neural implicit shape representations) 进行优化。

这些方法也许可以产生复杂的高清立体图像，但是他们目前只局限于一些简单的形状，以至于过平滑渲染 (oversmoothed renderings)

* 比较流行的方法： 使用 mesh-based representations of scenes with either diffuse or view-dependent appearance.
Differentiable rasterizers or pathtracers can directly optimize mesh representations to reproduce a set of input images using gradient descent.

这个方法不好，因为这个方法需要 a template mesh with fixed topology to be provided as an initialization before optimization, which is typically unavailable for unconstrained real-world scenes.


* 另外一种方法使用 volumetric representations to address the task of high-quality photorealistic view synthesis from a set of input RGB images. Volumetric 方法可以应用到一些复杂的形状和材料，同时也很适合gradient-based优化。然后作者表明了这些现有的volumetric 方法受到的局限性很大 by poor time and space complexity due to their discrete sampling.渲染高清的图片需要对三维空间更好的采样。

* 最后作者说了自己的方法来规避volumetric方法中的问题：We circumvent this problem by instead encoding a continuous volume within the parameters of a deep fully-connected neural network, which not only produces significantly higher quality renderings than prior volumetric approaches, but also requires just a fraction of the storage cost of those sampled volumetric representations. 


## 核心方法：

利用一个5D函数来表示一个静止的场景(scene). 这个5D函数可以输出在空间中每一个带有(x, y, z)坐标的点的每一个(θ, φ) 方向上散发的光，还有在每一个点上光线穿过的密度 (density)。

本论文运用了 deep fully-connected neural network 或者被称为 MLP multilayer perceptron，通过回归一个单独的5D数据 (x, y, z, θ, φ)，和其对应的一个volume density，还有其视角对应 (view-dependent) 的RGB颜色，来实现这个模型。


关键词: 5D数据 (x, y, z, θ, φ) , volume density, view-dependent RGB color


## 方法步骤：

1.  march camera rays through the scene to generate a sampled set of 3D points

2. use those points and their corresponding 2D viewing directions as input to the neural network to produce an output set of colors and densities

3. use classical volume rendering techniques to accumulate those colors and densities into a 2D image

### This process is naturally differentiable

Because this process is naturally differentiable, we can use gradient descent to optimize this model by minimizing the error between each observed image and the corresponding views rendered from our representation.



## Neural Radiance Field Scene Representation

论文通过一个5D矢量来表征一个连续的场景，其输入是之前提到过的三维空间位置 x = (x, y, z)，还有二维的视角方向(θ, φ) ，其输出是散发出的颜色 c = (r, g, b)，还有(volume density)容量密度 σ

如果FΘ是MLP网络，x = (x, y, z)坐标，视角 d 为3维笛卡尔坐标(cartisan unit vector)，那么整个过程可以看做公式化为：FΘ : (x, d) → (c, σ)

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf1.png">

论文为了让物体表现为multiview consistent，通过限制网络仅通过坐标x来预测volume density σ 而同时又允许RGB颜色 c 可以被通过坐标和视角来预测. MLP网络首先接受输入x和8层全连接层 (使用ReLU激活函数和每一层256个channels) , 然后输出volume density σ和256维的feature vector. 这个feature vector随后产生于通过拼接摄像机视角和通过一层额外的能够输出基于视角的RGB颜色的全连接层 (使用ReLU和128个channels)。



## Volume Rendering with Radiance Fields

Volume density σ(x) 可以被解释为一束光线停止于在位置x的一个无穷小的粒子上。 期望得到的颜色 C(r) 在摄像机的视线上 r(t) = o + td 的远近边界 tn 和 tf 有下列关系式:
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf2.png">


T(t) 代表着一条线上累计的光，从 tn 传输到 t. 比如，光线从tn 到 t 不碰到任何粒子的概率。需要估计通过所需虚拟相机的每个像素追溯的相机光线的积分C（r）。

我们对这样的连续积分使用Quadrature求积法. Deterministic quadrature (quadrature求定积分), 通常被用做渲染离散化的三位像素网络，会有效的影响我们个体表现的分辨率，因为 MLP 只会在应用在一组固定的且离散的坐标位置。

反之，我们使用分层采样 (stratified sampling) 的方法， 这样我们可以把 [tn, tf] 分成 N 个均等区间的分箱 (bins)， 然后统一随机在每一个分箱内抽取一个样本。

Quadrature求积法介绍：https://zhuanlan.zhihu.com/p/90607361

<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf3.png">

尽管我们使用一组离散的样本来计算这个积分，分层采样 (stratified sampling) 允许我们展现连续的场景，因为在整个过程中都是连续的位置(positions)来通过MLP网络。我们使用这些样本配合在Max文章的volume rendering review中提到的quadrature rule 来计算 C(r)。

注： Max, N.: Optical models for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics (1995)


<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf4.png">

上述公式中，其中<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf5.png" width="100" height="30"> 是代表邻近样本之间的距离，这个从集合 (ci, σi) 中计算 Cˆ(r) 的公式是可以很容易微分的并且

减少到使用alpha值 αi = 1 − exp(−σiδi).进行传统alpha合成


## Optimizing a Neural Radiance Field
作者一共提出了两种方法来优化 Neural Radiance Field，第一种是对于输入坐标的Positional encoding，第二个是hierarchical sampling，这个方法可以让我们更高效的采样高频样本。


### Positional encoding
作者发现网络如果直接输入xyzθφ 坐标，训练出来的物体渲染效果会非常不好。这和Rahaman的论文中描述得一致，原因是神经网络会倾向于学习低频的函数。他们也证明了在把输入放入网络中之前，使用高频函数把输入映射到高维空间然后再输入到网络中，网络能够更好的拟合包含高频函数变体的数据。

本文中，作者对这些发现进行了补充，通过重新组合两个函数 FΘ = F'Θ ◦ γ ， 来构建FΘ。其中一个是函数是学习到的，γ 是把 R 映射到高维空间 R2L，F'Θ 仍然是简单的MLP。

整个编码函数可以表示为下：
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf6.png">

文中采用的是用sin-cos函数，先将space坐标归一化到[-1,1]，然后采用sin-cos函数进行编码，本工作将3维position通过这种方式升到10维，方向升到4维。

另外一个比较类似的映射方法就是大名鼎鼎的transformer架构。然而transformer的目标却不一样，transformer把它用来做成一系列的带有离散位置信息的token，这种架构不包含顺序，次序的概念。反之，本文使用这些函数来映射连续的输入坐标到高维空间，使得 MLP 网络能够更好的拟合一个高频率的函数。

### Hierarchical volume sampling

对于一个场景，很显然各个位置的重要性是不一样的，有的地方是大片相同的内容（或者空的），有些地方则细节很多，对于全空间采用相同密度的采样，对于训练显然是不划算的。因此，作者提出coarse+fine的想法，作者训练了两个network，先训练一个coarse sample的网络，然后根据coarse网络的输出，再进行更加细致的采样。

我们首先使用分层采样来采样第一组位置记为 Nc, 然后使用 'coarse' 网络来拟合这些数据 (就像Volume Rendering with Radiance Fields中的两个公式)。通过这个 'coarse' 网络，然后，我们产生了一个沿着每条光线的每一个更精确的点采样，生成一个偏向于volume的相关部分的样本。为了实现这个方法，作者首先重写了alpha composited color使用上面求C(r)的公式来求Cˆc(r) 作为这条光线上的所有采样过的颜色 Ci 的加权和。
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf7.png">

通过归一化这些权重为<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf8.png" width="140" height="30">沿着光线产生了一个包含分段常数(piecewised-constant) 的概率密度函数（简称PDF:Probability Density Function)。作者从这个分布中使用 inverse transform sampling 采样了第二个有 Nf 个位置的集合。使用第一个样本集合和第二个样本集合的并集来训练我们的 'fine' 网络，然后同样使用上面求 C(r) 的公式来计算最后这条光线上渲染的颜色 Cˆf (r) 。不一样的是，这一次我们使用了所有 Nc + Nf 的样本。这个方法分配了更多的样本到一些我们期望有可见物体的区域。

同样这个方法也达成了和importance sampling一样的目标，但是我们使用这些采样的值作为整个积分域的非均匀的离散化而不是把每一个样本作为这整个积分的一个独立概率估算。


## Implementation details

每一个场景都是用的分别的neural continuous volume representation network。 这样只需要一个捕捉到的场景RGB图片，比如：对应的摄像头视角，其本身的参数，和场景边界。

(作者使用ground truth的合成数据摄像头视角，本身参数，和边界，还有COLMAP structure-from-motion package来拟合真实数据的参数。)

在每一个训练迭代次数，作者从整个数据集里随机采样了一个batch的摄像头机光线，然后使用hierarchical sampling 从 coarse 网络来产生 Nc 个样本，最后从fine 网络中产生 Nc + Nf 个样本。接下来就可以使用 Volume Rendering with Radiance Fields中所描述的过程来渲染两个样本集中每束光线的颜色。

我们loss 函数就是全部的二次方误差，取自于 coarse和 fine 渲染的颜色和真实的像素颜色差：
<img src="https://github.com/xiaoxingchen505/SOA_Deep_Learning/blob/main/images/nerf9.png">

上面的公式中，R是每个batch中的光线的集合，C(r), Cˆc(r) , 和 Cˆf (r) 分别是对于每一个光线 r 的ground truth， 预测的 coarse volume RGB颜色，预测的 fine volume RGB颜色。

注意：尽管最后的渲染结果出自于 Cˆf (r)，我们也同样最小化了 Cˆc(r) 的loss，所以coarse网络中的权重分配可以被用来分配样本到fine网络里。

## 论文总结：

优点，贡献：
1. 模型简单，使用basic MLP network
2. 克服了在高分辨率下对复杂场景建模时，离散体素网格难以承受的存储成本。
3. 提出一个positional encoding 来把每一个5D 坐标输入映射为更高的空间维度。
4. 一个基于传统volume rendering技术的可微分 (differentiable) 过程