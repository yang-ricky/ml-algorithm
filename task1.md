## 任务1：感知器实现

### 学习目标

* [ ] 理解感知器模型与线性可分假设
* [ ] 掌握感知器学习规则及收敛定理
* [ ] 能够评估训练误差与泛化误差

### 任务描述

* [ ] 创建 `PerceptronClassifier` 类，实现在线学习与批量更新
* [ ] 编写单元测试类 `IrisBinaryPerceptronTest`，区分鸢尾花两类

### 场景

* [ ] 使用鸢尾花 (setosa vs. versicolor) 特征，预测花类别

### 产出要求

* [ ] 核心算法类：PerceptronClassifier.java
* [ ] 单元测试类：IrisBinaryPerceptronTest.java
* [ ] 测试数据集：iris\_binary.csv

---

## 任务2：多层感知器实现

### 学习目标

* [ ] 理解多层感知器非线性表示能力
* [ ] 掌握隐藏层/激活函数设计
* [ ] 能够实现前向与反向传播框架

### 任务描述

* [ ] 创建 `MultilayerPerceptron` 类，支持任意隐藏层与 ReLU、Tanh
* [ ] 编写单元测试类 `MNISTDigitMLPTest`，识别手写数字

### 场景

* [ ] MNIST 784 维像素输入 → 10 类数字分类

### 产出要求

* [ ] 核心算法类：MultilayerPerceptron.java
* [ ] 单元测试类：MNISTDigitMLPTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务3：反向传播算法可视化

### 学习目标

* [ ] 彻底理解链式法则与梯度流
* [ ] 学会利用梯度检查验证实现正确性
* [ ] 能够可视化梯度消失/爆炸现象

### 任务描述

* [ ] 创建 `BackpropVisualizer` 工具类，支持梯度检查与热力图输出
* [ ] 编写单元测试类 `XORBackpropCheckTest`

### 场景

* [ ] 经典 XOR 数据集，展示训练早期梯度分布

### 产出要求

* [ ] 核心工具类：BackpropVisualizer.java
* [ ] 单元测试类：XORBackpropCheckTest.java
* [ ] 测试数据集：xor.csv

---

## 任务4：自组织特征映射（SOM）实现

### 学习目标

* [ ] 理解竞争学习与邻域函数
* [ ] 掌握二维拓扑映射原理
* [ ] 能够用 U-Matrix 评估聚类可视化

### 任务描述

* [ ] 创建 `SelfOrganizingMap` 类，支持矩形拓扑
* [ ] 编写单元测试类 `ColorQuantizationSOMTest`

### 场景

* [ ] 通过 SOM 对 24-bit RGB 颜色进行 16 色量化

### 产出要求

* [ ] 核心算法类：SelfOrganizingMap.java
* [ ] 单元测试类：ColorQuantizationSOMTest.java
* [ ] 测试数据集：rgb\_pixels.csv

---

## 任务5：深度前馈网络——隐藏层设计

### 学习目标

* [ ] 掌握隐藏层宽度/深度对表示能力的影响
* [ ] 学会使用批归一化加速收敛

### 任务描述

* [ ] 创建 `HiddenLayerDesigner` 辅助类，自动推荐层宽
* [ ] 编写单元测试类 `HousePriceHiddenLayerTest`，对比不同配置

### 场景

* [ ] 波士顿房价回归，评估 MAE 与训练时间

### 产出要求

* [ ] 辅助类：HiddenLayerDesigner.java
* [ ] 单元测试类：HousePriceHiddenLayerTest.java
* [ ] 测试数据集：boston\_housing.csv

---

## 任务6：深度前馈网络——输出层设计

### 学习目标

* [ ] 熟悉 Softmax、Sigmoid、Linear 输出的适用场景
* [ ] 理解交叉熵与均方误差的差异

### 任务描述

* [ ] 扩展 `MultilayerPerceptron`，根据任务类型自动选择输出层
* [ ] 编写单元测试类 `OutputLayerChoiceTest`，对比分类 / 回归

### 场景

* [ ] CIFAR-10 分类与房价回归双任务对比

### 产出要求

* [ ] 更新 MultilayerPerceptron.java
* [ ] 单元测试类：OutputLayerChoiceTest.java
* [ ] 测试数据集：cifar10\_subset.csv, boston\_housing.csv

---

## 任务7：深度前馈网络在回归任务中的应用

### 学习目标

* [ ] 掌握网络正则与回归平滑
* [ ] 了解输出不确定度估计

### 任务描述

* [ ] 创建 `MLPRegressor` 类，使用线性输出 + MSE
* [ ] 编写单元测试类 `EnergyEfficiencyRegressionTest`

### 场景

* [ ] 建筑能耗数据 → 预测取暖负荷

### 产出要求

* [ ] 核心算法类：MLPRegressor.java
* [ ] 单元测试类：EnergyEfficiencyRegressionTest.java
* [ ] 测试数据集：energy\_efficiency.csv

---

## 任务8：数据增强正则化

### 学习目标

* [ ] 掌握图像随机裁剪、旋转、颜色抖动
* [ ] 理解数据增强对过拟合的抑制

### 任务描述

* [ ] 创建 `ImageDataAugmenter` 工具类，链式配置多种变换
* [ ] 编写单元测试类 `CatDogAugmentTest`

### 场景

* [ ] 1 k 张猫狗照片扩增至 10 k 张用于分类

### 产出要求

* [ ] 工具类：ImageDataAugmenter.java
* [ ] 单元测试类：CatDogAugmentTest.java
* [ ] 原始数据集：cat\_dog\_1k/

---

## 任务9：Dropout 正则化

### 学习目标

* [ ] 理解 Dropout 的集成解释
* [ ] 掌握训练/推理阶段的区别

### 任务描述

* [ ] 在 `MultilayerPerceptron` 中加入可配置 Dropout 层
* [ ] 编写单元测试类 `MNISTDropoutTest`，比较过拟合情况

### 场景

* [ ] 手写数字 5-epoch vs. 50-epoch 对照实验

### 产出要求

* [ ] 更新 MultilayerPerceptron.java
* [ ] 单元测试类：MNISTDropoutTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务10：架构级正则化——网络剪枝

### 学习目标

* [ ] 理解参数稀疏化与 FLOPs 降低
* [ ] 掌握迭代剪枝与微调流程

### 任务描述

* [ ] 创建 `NetworkPruner` 类，实现权重阈值剪枝
* [ ] 编写单元测试类 `PrunedLeNetTest`

### 场景

* [ ] 对 LeNet 剪枝 50%，比较推理时延

### 产出要求

* [ ] 核心算法类：NetworkPruner.java
* [ ] 单元测试类：PrunedLeNetTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务11：误差函数正则化——标签平滑

### 学习目标

* [ ] 理解标签平滑减少过拟合原理
* [ ] 掌握超参数 ε 调节

### 任务描述

* [ ] 创建 `LabelSmoothingCrossEntropy` 损失类
* [ ] 编写单元测试类 `CIFARLabelSmoothingTest`

### 场景

* [ ] CIFAR-10 训练 ResNet-18，对比 top-1 准确率

### 产出要求

* [ ] 损失类：LabelSmoothingCrossEntropy.java
* [ ] 单元测试类：CIFARLabelSmoothingTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务12：正则化项（L1/L2）对比实验

### 学习目标

* [ ] 掌握权重衰减与稀疏化效果
* [ ] 理解 λ 超参数搜索

### 任务描述

* [ ] 创建 `RegularizationTermExperiment` 脚本
* [ ] 编写单元测试类 `RegularizationMNISTTest`

### 场景

* [ ] MLP 在 MNIST 上测试 L1 vs. L2

### 产出要求

* [ ] 实验脚本：RegularizationTermExperiment.java
* [ ] 单元测试类：RegularizationMNISTTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务13：初始化正则化——权重初始化策略

### 学习目标

* [ ] 理解 Xavier 与 He 初始化差异
* [ ] 评估不同初始化对收敛速度影响

### 任务描述

* [ ] 创建 `WeightInitializer` 枚举（RANDOM, XAVIER, HE）
* [ ] 编写单元测试类 `InitializationSpeedTest`

### 场景

* [ ] 在 CIFAR-10 上训练小型 CNN，记录前 10 epoch 损失

### 产出要求

* [ ] 工具类：WeightInitializer.java
* [ ] 单元测试类：InitializationSpeedTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务14：参数更新正则化——权重衰减实现

### 学习目标

* [ ] 理解权重衰减等价于 L2 正则
* [ ] 掌握实现方式（优化器内部乘系数）

### 任务描述

* [ ] 在 `SGDOptimizer` 中加入 `weightDecay` 参数
* [ ] 编写单元测试类 `SGDWeightDecayTest`

### 场景

* [ ] 对比无衰减 vs. 衰减 λ=1e-4 的 CIFAR-10 训练

### 产出要求

* [ ] 更新 SGDOptimizer.java
* [ ] 单元测试类：SGDWeightDecayTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务15：早停（终止条件正则化）

### 学习目标

* [ ] 理解验证集监控与 patience 概念
* [ ] 能够防止过拟合并节省训练时间

### 任务描述

* [ ] 创建 `EarlyStoppingCallback` 类
* [ ] 编写单元测试类 `EarlyStoppingFashionMNISTTest`

### 场景

* [ ] Fashion-MNIST 训练 MLP，观察早停点与最佳准确率

### 产出要求

* [ ] 回调类：EarlyStoppingCallback.java
* [ ] 单元测试类：EarlyStoppingFashionMNISTTest.java
* [ ] 测试数据集：fashion\_mnist.csv

---

## 任务16：马尔可夫决策过程建模

### 学习目标

* [ ] 理解 MDP (S, A, P, R, γ) 定义
* [ ] 掌握价值迭代与策略迭代

### 任务描述

* [ ] 创建 `GridWorldMDP` 类，支持 4 向移动
* [ ] 编写单元测试类 `GridWorldPolicyIterationTest`

### 场景

* [ ] 4×4 走格子游戏求最优策略

### 产出要求

* [ ] 核心类：GridWorldMDP.java
* [ ] 单元测试类：GridWorldPolicyIterationTest.java
* [ ] 测试关卡文件：gridworld\_4x4.txt

---

## 任务17：无监督强化辅助学习

### 学习目标

* [ ] 掌握基于预测误差的内在奖励
* [ ] 了解 Curiosity-driven 学习思想

### 任务描述

* [ ] 创建 `CuriosityAgent` 类，利用预测下一个状态误差做奖励
* [ ] 编写单元测试类 `MountainCarCuriosityTest`

### 场景

* [ ] OpenAI Gym MountainCar，无外部奖励也能学习前进

### 产出要求

* [ ] 核心算法类：CuriosityAgent.java
* [ ] 单元测试类：MountainCarCuriosityTest.java
* [ ] 环境日志：mountaincar\_log.csv

---

## 任务18：Q-学习算法实现

### 学习目标

* [ ] 理解 Q-Learning 更新公式
* [ ] 掌握 ε-贪婪与学习率调度

### 任务描述

* [ ] 创建 `QLearningAgent` 类
* [ ] 编写单元测试类 `TaxiDriverQTest`

### 场景

* [ ] Taxi-v3 环境接客送客

### 产出要求

* [ ] 核心算法类：QLearningAgent.java
* [ ] 单元测试类：TaxiDriverQTest.java
* [ ] 环境日志：taxi\_episode.csv

---

## 任务19：卷积神经网络基础实现

### 学习目标

* [ ] 理解卷积、池化、通道概念
* [ ] 掌握参数共享与局部连接优势

### 任务描述

* [ ] 创建 `SimpleCNN` 类（Conv-Relu-Pool ×2 + FC）
* [ ] 编写单元测试类 `CIFAR10SimpleCNNTest`

### 场景

* [ ] CIFAR-10 分类，比较与 MLP 准确率

### 产出要求

* [ ] 核心算法类：SimpleCNN.java
* [ ] 单元测试类：CIFAR10SimpleCNNTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务20：残差网络（ResNet）实现

### 学习目标

* [ ] 理解恒等映射与梯度流畅通
* [ ] 掌握残差块结构

### 任务描述

* [ ] 创建 `ResNet18` 类，实现 BasicBlock
* [ ] 编写单元测试类 `ResNetCIFARTest`

### 场景

* [ ] CIFAR-10 训练 ResNet-18, 比较与 SimpleCNN

### 产出要求

* [ ] 核心算法类：ResNet18.java
* [ ] 单元测试类：ResNetCIFARTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务21：生成式对抗网络实现

### 学习目标

* [ ] 理解 Generator / Discriminator 对抗框架
* [ ] 掌握判别器与生成器交替训练

### 任务描述

* [ ] 创建 `DCGAN` 类，基于卷积结构生成 28×28 图像
* [ ] 编写单元测试类 `MNISTDCGANTest`，可视化生成样本

### 场景

* [ ] MNIST 生成手写数字照片

### 产出要求

* [ ] 核心算法类：DCGAN.java
* [ ] 单元测试类：MNISTDCGANTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务22：径向基函数神经网络实现

### 学习目标

* [ ] 理解 RBF 隐层与高斯基函数
* [ ] 掌握 K-Means 选中心与线性输出权重

### 任务描述

* [ ] 创建 `RBFNetwork` 类
* [ ] 编写单元测试类 `RBFIrisTest`

### 场景

* [ ] 鸢尾花三类分类

### 产出要求

* [ ] 核心算法类：RBFNetwork.java
* [ ] 单元测试类：RBFIrisTest.java
* [ ] 测试数据集：iris.csv

---

## 任务23：模糊神经网络实现

### 学习目标

* [ ] 掌握模糊规则与神经网络结合
* [ ] 理解 Sugeno-型推理

### 任务描述

* [ ] 创建 `FuzzyNeuroController` 类，控制空调温度
* [ ] 编写单元测试类 `AirConditionFuzzyNNTest`

### 场景

* [ ] 输入室温/湿度 → 输出压缩机功率

### 产出要求

* [ ] 核心算法类：FuzzyNeuroController.java
* [ ] 单元测试类：AirConditionFuzzyNNTest.java
* [ ] 测试数据集：ac\_control.csv

---

## 任务24：局部最小值与鞍点可视化

### 学习目标

* [ ] 理解非凸损失景观
* [ ] 能识别鞍点陷阱

### 任务描述

* [ ] 创建 `LossLandscapeExplorer` 工具类，绘制 2D 切片
* [ ] 编写单元测试类 `MLPLossSurfaceTest`

### 场景

* [ ] 2-层 MLP 在 XOR 数据上的损失面

### 产出要求

* [ ] 工具类：LossLandscapeExplorer.java
* [ ] 单元测试类：MLPLossSurfaceTest.java
* [ ] 测试数据集：xor.csv

---

## 任务25：SGD 优化器基础实现

### 学习目标

* [ ] 理解随机梯度下降与批量梯度的差异
* [ ] 掌握学习率调度

### 任务描述

* [ ] 创建 `SGDOptimizer` 类，支持 step / cosine decay
* [ ] 编写单元测试类 `SGDIrisTest`

### 场景

* [ ] 鸢尾花 MLP 训练对比 batch = 32 vs. 全批量

### 产出要求

* [ ] 核心算法类：SGDOptimizer.java
* [ ] 单元测试类：SGDIrisTest.java
* [ ] 测试数据集：iris.csv

---

## 任务26：随机降低噪声（Gradient Noise Scale）

### 学习目标

* [ ] 掌握梯度噪声注入平滑优化
* [ ] 了解 GNS 公式评估噪声

### 任务描述

* [ ] 创建 `GradientNoiseInjector` 类，训练时添加高斯噪声
* [ ] 编写单元测试类 `NoiseInjectMNISTTest`

### 场景

* [ ] MNIST MLP 加噪声 vs. 不加噪声收敛曲线

### 产出要求

* [ ] 工具类：GradientNoiseInjector.java
* [ ] 单元测试类：NoiseInjectMNISTTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务27：动态采样批大小

### 学习目标

* [ ] 理解 batch size 对梯度方差的影响
* [ ] 掌握动态 batch 策略

### 任务描述

* [ ] 创建 `DynamicBatchSGD` 优化器，随 epoch 增大 batch
* [ ] 编写单元测试类 `DynamicBatchCIFARTest`

### 场景

* [ ] CIFAR-10 CNN，batch 32 → 128 → 256 动态调整

### 产出要求

* [ ] 优化器：DynamicBatchSGD.java
* [ ] 单元测试类：DynamicBatchCIFARTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务28：梯度聚合 (Gradient Accumulation)

### 学习目标

* [ ] 学会在显存有限时模拟大 batch
* [ ] 理解累积步骤与学习率关系

### 任务描述

* [ ] 创建 `GradientAccumulator` 包装器
* [ ] 编写单元测试类 `AccumulationTransformerTest`

### 场景

* [ ] 小显存 GPU 训练 Transformer，小批累积 4×

### 产出要求

* [ ] 包装类：GradientAccumulator.java
* [ ] 单元测试类：AccumulationTransformerTest.java
* [ ] 测试数据集：wmt14\_en\_de\_small.csv

---

## 任务29：迭代平均 (Polyak Averaging)

### 学习目标

* [ ] 理解权重移动平均平滑训练曲线
* [ ] 掌握 EMA 超参 β 选择

### 任务描述

* [ ] 创建 `WeightAverager` 类，实现滑动平均
* [ ] 编写单元测试类 `EMACIFARTest`

### 场景

* [ ] CIFAR-10 ResNet 使用 EMA，比较验证损失

### 产出要求

* [ ] 工具类：WeightAverager.java
* [ ] 单元测试类：EMACIFARTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务30：牛顿法优化

### 学习目标

* [ ] 理解二阶泰勒展开与 Hessian
* [ ] 掌握牛顿步与线搜索

### 任务描述

* [ ] 创建 `NewtonOptimizer` 类（小规模可逆 Hessian）
* [ ] 编写单元测试类 `LogRegNewtonTest`

### 场景

* [ ] 二分类逻辑回归在 IRIS 二类子集

### 产出要求

* [ ] 优化器：NewtonOptimizer.java
* [ ] 单元测试类：LogRegNewtonTest.java
* [ ] 测试数据集：iris\_binary.csv

---

## 任务31：高斯牛顿法优化

### 学习目标

* [ ] 掌握在非线性最小二乘中的应用
* [ ] 理解近似 Hessian 的计算

### 任务描述

* [ ] 创建 `GaussNewtonOptimizer` 类
* [ ] 编写单元测试类 `CurveFittingGaussNewtonTest`

### 场景

* [ ] 拟合非线性指数衰减曲线

### 产出要求

* [ ] 优化器：GaussNewtonOptimizer.java
* [ ] 单元测试类：CurveFittingGaussNewtonTest.java
* [ ] 测试数据集：exp\_decay.csv

---

## 任务32：Hessian-free 牛顿法

### 学习目标

* [ ] 理解共轭梯度近似 Hessian 乘积
* [ ] 适用于大规模深网

### 任务描述

* [ ] 创建 `HessianFreeOptimizer` 类
* [ ] 编写单元测试类 `HFCurvatureTest`

### 场景

* [ ] 2-层全连接网络在 MNIST 小批实验

### 产出要求

* [ ] 优化器：HessianFreeOptimizer.java
* [ ] 单元测试类：HFCurvatureTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务33：动量方法 (SGD+Momentum)

### 学习目标

* [ ] 理解动量加速收敛机理
* [ ] 掌握 β 参数调节

### 任务描述

* [ ] 创建 `MomentumOptimizer` 类
* [ ] 编写单元测试类 `MomentumFashionMNISTTest`

### 场景

* [ ] Fashion-MNIST MLP，比较无动量 vs. β=0.9

### 产出要求

* [ ] 优化器：MomentumOptimizer.java
* [ ] 单元测试类：MomentumFashionMNISTTest.java
* [ ] 测试数据集：fashion\_mnist.csv

---

## 任务34：加速下降方法 (Nesterov)

### 学习目标

* [ ] 理解 Nesterov 预取梯度思想
* [ ] 与 Momentum 的差异

### 任务描述

* [ ] 创建 `NesterovOptimizer` 类
* [ ] 编写单元测试类 `NesterovCIFARTest`

### 场景

* [ ] CIFAR-10 CNN 训练对比

### 产出要求

* [ ] 优化器：NesterovOptimizer.java
* [ ] 单元测试类：NesterovCIFARTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务35：坐标下降优化

### 学习目标

* [ ] 理解分块更新思想
* [ ] 掌握在稀疏问题中的优势

### 任务描述

* [ ] 创建 `CoordinateDescentOptimizer` 类
* [ ] 编写单元测试类 `LassoCoordinateDescentTest`

### 场景

* [ ] 高维 LASSO 回归 (1 万特征)

### 产出要求

* [ ] 优化器：CoordinateDescentOptimizer.java
* [ ] 单元测试类：LassoCoordinateDescentTest.java
* [ ] 测试数据集：lasso\_highdim.csv

---

## 任务36：欠完备自编码器实现

### 学习目标

* [ ] 理解压缩编码与瓶颈维度
* [ ] 掌握重建误差评估

### 任务描述

* [ ] 创建 `UndercompleteAutoencoder` 类
* [ ] 编写单元测试类 `MNISTAutoencoderTest`

### 场景

* [ ] 将 784 维 MNIST 压缩到 32 维重建

### 产出要求

* [ ] 核心算法类：UndercompleteAutoencoder.java
* [ ] 单元测试类：MNISTAutoencoderTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务37：过完备自编码器实现

### 学习目标

* [ ] 理解过完备表示与正则化需求
* [ ] 掌握权重噪声抑制

### 任务描述

* [ ] 创建 `OvercompleteAutoencoder` 类，隐藏维度 1024
* [ ] 编写单元测试类 `OvercompleteDenoiseTest`

### 场景

* [ ] 对 Fashion-MNIST 图片过完备编码并加噪声

### 产出要求

* [ ] 核心算法类：OvercompleteAutoencoder.java
* [ ] 单元测试类：OvercompleteDenoiseTest.java
* [ ] 测试数据集：fashion\_mnist.csv

---

## 任务38：稀疏自编码器实现

### 学习目标

* [ ] 理解 KL 散度稀疏约束
* [ ] 掌握稀疏目标 ρ 与 β

### 任务描述

* [ ] 创建 `SparseAutoencoder` 类
* [ ] 编写单元测试类 `SparseAEFacesTest`

### 场景

* [ ] 用于面部特征提取 (ORL-Faces)

### 产出要求

* [ ] 核心算法类：SparseAutoencoder.java
* [ ] 单元测试类：SparseAEFacesTest.java
* [ ] 测试数据集：orl\_faces.csv

---

## 任务39：变分自编码器实现

### 学习目标

* [ ] 理解重参数化技巧
* [ ] 掌握 KL 散度损失计算

### 任务描述

* [ ] 创建 `VariationalAutoencoder` 类
* [ ] 编写单元测试类 `MNISTVAEGenerateTest`

### 场景

* [ ] MNIST VAE 生成手写数字样本

### 产出要求

* [ ] 核心算法类：VariationalAutoencoder.java
* [ ] 单元测试类：MNISTVAEGenerateTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务40：去噪自编码器实现

### 学习目标

* [ ] 理解噪声鲁棒表示
* [ ] 掌握高斯/遮挡噪声注入

### 任务描述

* [ ] 创建 `DenoisingAutoencoder` 类
* [ ] 编写单元测试类 `ImageDenoiseAETest`

### 场景

* [ ] 对含噪 Fashion-MNIST 恢复干净图像

### 产出要求

* [ ] 核心算法类：DenoisingAutoencoder.java
* [ ] 单元测试类：ImageDenoiseAETest.java
* [ ] 测试数据集：fashion\_mnist\_noisy.csv

---

## 任务41：深度信念网络实现

### 学习目标

* [ ] 理解逐层无监督预训练
* [ ] 掌握 RBM 堆叠

### 任务描述

* [ ] 创建 `DeepBeliefNetwork` 类
* [ ] 编写单元测试类 `DBNMNISTTest`

### 场景

* [ ] 对 MNIST 进行无监督预训练再微调

### 产出要求

* [ ] 核心算法类：DeepBeliefNetwork.java
* [ ] 单元测试类：DBNMNISTTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务42：受限玻尔兹曼机（RBM）实现

### 学习目标

* [ ] 理解能量函数与对比散度 CD-k
* [ ] 掌握二元–高斯 RBM

### 任务描述

* [ ] 创建 `RestrictedBoltzmannMachine` 类
* [ ] 编写单元测试类 `RBMMNISTTest`

### 场景

* [ ] 对 MNIST 进行特征提取，比较重建误差

### 产出要求

* [ ] 核心算法类：RestrictedBoltzmannMachine.java
* [ ] 单元测试类：RBMMNISTTest.java
* [ ] 测试数据集：mnist\_train.csv

---

## 任务43：循环神经网络实现

### 学习目标

* [ ] 理解时间展开与梯度传播
* [ ] 掌握截断 BPTT

### 任务描述

* [ ] 创建 `SimpleRNN` 类
* [ ] 编写单元测试类 `TemperatureSequenceRNNTest`

### 场景

* [ ] 预测每日气温序列的下一天

### 产出要求

* [ ] 核心算法类：SimpleRNN.java
* [ ] 单元测试类：TemperatureSequenceRNNTest.java
* [ ] 测试数据集：daily\_temperature.csv

---

## 任务44：长短期记忆网络（LSTM）实现

### 学习目标

* [ ] 理解门控机制解决梯度消失
* [ ] 掌握多层 LSTM 叠加

### 任务描述

* [ ] 创建 `LSTMNetwork` 类
* [ ] 编写单元测试类 `IMDBSentimentLSTMTest`

### 场景

* [ ] IMDB 长文本情感分析

### 产出要求

* [ ] 核心算法类：LSTMNetwork.java
* [ ] 单元测试类：IMDBSentimentLSTMTest.java
* [ ] 测试数据集：imdb\_sentiment.csv

---

## 任务45：Transformer 模型实现

### 学习目标

* [ ] 理解自注意力与位置编码
* [ ] 掌握编码器–解码器结构

### 任务描述

* [ ] 创建 `MiniTransformer` 类（2 层 encoder-decoder）
* [ ] 编写单元测试类 `EnDeTranslationTest`

### 场景

* [ ] 100 k 条英德平行句子小数据集翻译

### 产出要求

* [ ] 核心算法类：MiniTransformer.java
* [ ] 单元测试类：EnDeTranslationTest.java
* [ ] 测试数据集：wmt14\_en\_de\_small.csv

---

## 任务46：自注意力机制可视化

### 学习目标

* [ ] 理解 Query/Key/Value 点积注意力
* [ ] 能够解释头部关注位置

### 任务描述

* [ ] 创建 `AttentionHeatmap` 工具类，可导出图像
* [ ] 编写单元测试类 `AttentionVisualizationTest`

### 场景

* [ ] 对 “The quick brown fox…” 句子显示注意力矩阵

### 产出要求

* [ ] 工具类：AttentionHeatmap.java
* [ ] 单元测试类：AttentionVisualizationTest.java
* [ ] 输入文本：attention\_example.txt

---

## 任务47：Transformer 解码器文本生成

### 学习目标

* [ ] 掌握自回归解码与 beam search
* [ ] 学会温度采样控制多样性

### 任务描述

* [ ] 创建 `TransformerDecoderGenerator` 类
* [ ] 编写单元测试类 `StoryGenerationTest`

### 场景

* [ ] 以童话开头句子生成后续 100 字

### 产出要求

* [ ] 核心算法类：TransformerDecoderGenerator.java
* [ ] 单元测试类：StoryGenerationTest.java
* [ ] 训练语料：fairy\_tale\_corpus.txt

---

## 任务48：伊辛模型能量最小化模拟

### 学习目标

* [ ] 理解自旋相互作用与格点模型
* [ ] 掌握 Metropolis 更新

### 任务描述

* [ ] 创建 `IsingModelSimulator` 类
* [ ] 编写单元测试类 `IsingPhaseTransitionTest`

### 场景

* [ ] 20×20 2D 格子在不同温度下的磁化率

### 产出要求

* [ ] 核心算法类：IsingModelSimulator.java
* [ ] 单元测试类：IsingPhaseTransitionTest.java
* [ ] 初始配置文件：ising\_init.txt

---

## 任务49：量子退火原理演示

### 学习目标

* [ ] 理解量子隧穿与退火对比
* [ ] 掌握模拟量子退火步骤

### 任务描述

* [ ] 创建 `QuantumAnnealingSolver` 类（路径积分蒙特卡罗近似）
* [ ] 编写单元测试类 `QATSPTest`

### 场景

* [ ] 给定 20 城 TSP，比较 QA vs. 模拟退火结果

### 产出要求

* [ ] 核心算法类：QuantumAnnealingSolver.java
* [ ] 单元测试类：QATSPTest.java
* [ ] 测试数据集：tsp20.csv

---

## 任务50：重整化群 (RG) 可视化

### 学习目标

* [ ] 理解尺度变换与临界指数
* [ ] 掌握 Kadanoff 块自旋思想

### 任务描述

* [ ] 创建 `RenormalizationVisualizer` 类，对 1D Ising 作 RG 迭代
* [ ] 编写单元测试类 `RGFlowTest`

### 场景

* [ ] 演示耦合常数流向固定点

### 产出要求

* [ ] 工具类：RenormalizationVisualizer.java
* [ ] 单元测试类：RGFlowTest.java
* [ ] 输入文件：ising\_1d\_params.json

---

## 任务51：布朗运动路径模拟

### 学习目标

* [ ] 掌握 Wiener 过程生成
* [ ] 理解均方位移随时间增长

### 任务描述

* [ ] 创建 `BrownianMotionSimulator` 类
* [ ] 编写单元测试类 `BrownianMSDTest`

### 场景

* [ ] 模拟花粉颗粒随机运动并绘制轨迹

### 产出要求

* [ ] 核心算法类：BrownianMotionSimulator.java
* [ ] 单元测试类：BrownianMSDTest.java
* [ ] 输出数据：brownian\_paths.csv

---

## 任务52：随机游走与 PageRank 实验

### 学习目标

* [ ] 理解随机游走稳态分布
* [ ] 掌握 PageRank 迭代公式

### 任务描述

* [ ] 创建 `RandomWalkPageRank` 类，实现随机跳转 α
* [ ] 编写单元测试类 `WebGraphPageRankTest`

### 场景

* [ ] 对 100 页小型网站计算排名

### 产出要求

* [ ] 核心算法类：RandomWalkPageRank.java
* [ ] 单元测试类：WebGraphPageRankTest.java
* [ ] 图结构文件：web\_graph.edgelist

---

## 任务53：蚁群算法路径优化

### 学习目标

* [ ] 理解信息素更新与启发式函数
* [ ] 掌握蒸发率与强化系数调节

### 任务描述

* [ ] 创建 `AntColonyPathPlanner` 类
* [ ] 编写单元测试类 `WarehousePathACOTest`

### 场景

* [ ] 仓库机器人从入口到货架最短路径

### 产出要求

* [ ] 核心算法类：AntColonyPathPlanner.java
* [ ] 单元测试类：WarehousePathACOTest.java
* [ ] 地图文件：warehouse\_grid.txt

---

## 任务54：元细胞自动机实验

### 学习目标

* [ ] 理解元胞自动机局域规则
* [ ] 掌握 Game of Life 动态

### 任务描述

* [ ] 创建 `CellularAutomaton` 类
* [ ] 编写单元测试类 `GameOfLifePatternTest`

### 场景

* [ ] 生成 glider 航迹并统计周期性

### 产出要求

* [ ] 核心算法类：CellularAutomaton.java
* [ ] 单元测试类：GameOfLifePatternTest.java
* [ ] 初始模式：glider.txt

---

## 任务55：沙堆模型自组织临界性

### 学习目标

* [ ] 理解局部放沙导致全局雪崩分布
* [ ] 掌握幂律分布分析

### 任务描述

* [ ] 创建 `SandpileModelSimulator` 类
* [ ] 编写单元测试类 `SandpileAvalancheTest`

### 场景

* [ ] 50×50 格子投沙，记录雪崩规模

### 产出要求

* [ ] 核心算法类：SandpileModelSimulator.java
* [ ] 单元测试类：SandpileAvalancheTest.java
* [ ] 输出数据：sandpile\_avalanches.csv

---

## 任务56：残差网络优化对比实验

### 学习目标

* [ ] 比较 ResNet 各种优化器 (SGD, Adam, Nesterov)
* [ ] 评估收敛速度与最终准确率

### 任务描述

* [ ] 创建 `ResNetOptimizerBenchmark` 脚本
* [ ] 编写单元测试类 `ResNetOptimizerBenchmarkTest`

### 场景

* [ ] CIFAR-10 ResNet-18 三优化器对比

### 产出要求

* [ ] 脚本：ResNetOptimizerBenchmark.java
* [ ] 单元测试类：ResNetOptimizerBenchmarkTest.java
* [ ] 测试数据集：cifar10\_subset.csv

---

## 任务57：优化器调度器 (One-Cycle)

### 学习目标

* [ ] 理解 One-Cycle 学习率策略
* [ ] 掌握实现 cosine-annealing + momentum up/down

### 任务描述

* [ ] 创建 `OneCycleScheduler` 类，可插入任意 Optimizer
* [ ] 编写单元测试类 `OneCycleSchedulerTest`

### 场景

* [ ] 训练 Transformer 小模型 10 epoch 对比固定 lr

### 产出要求

* [ ] 调度器：OneCycleScheduler.java
* [ ] 单元测试类：OneCycleSchedulerTest.java
* [ ] 测试数据集：wmt14\_en\_de\_small.csv
