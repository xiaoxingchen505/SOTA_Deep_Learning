

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

论文为了让物体表现为multiview consistent，通过限制网络仅通过坐标x来预测volume density σ 而同时又允许RGB颜色c又可以被通过坐标和视角来预测. MLP网络首先接受输入x和8层全连接层 (使用ReLU激活函数和每一层256个channels), 然后输出volume density σ和256维的feature vector. 这个feature vector随后产生于通过拼接摄像机视角和通过一层额外的能够输出基于视角的RGB颜色的全连接层(使用ReLU和128个channels)。


## 论文总结：

优点，贡献：
1. 模型简单，使用basic MLP network
2. 克服了在高分辨率下对复杂场景建模时，离散体素网格难以承受的存储成本。
3. 提出一个positional encoding 来把每一个5D 坐标输入映射为更高的空间维度。
4. 一个基于传统volume rendering技术的可微分 (differentiable) 过程