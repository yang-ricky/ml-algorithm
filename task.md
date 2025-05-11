## 任务1：一元线性回归实现

### 学习目标
- [ ] 理解一元线性回归的假设与最小二乘估计
- [ ] 掌握回归系数的封闭解与梯度下降求解
- [ ] 能够用残差分析评估拟合优劣

### 任务描述
- [ ] 创建 UnivariateLinearRegression 类，实现闭式解与批量梯度下降两种训练方法
- [ ] 编写单元测试类 HousePriceSimpleTest，验证面积-价格公开数据集的拟合精度

### 场景
- [ ] 使用北京二手房 (面积, 价格) CSV 数据，预测给定面积的房价

### 产出要求
- [ ] 核心算法类：UnivariateLinearRegression.java
- [ ] 单元测试类：HousePriceSimpleTest.java
- [ ] 测试数据集：house_price_simple.csv

---

## 任务2：多元线性回归实现

### 学习目标
- [ ] 理解多元线性回归与矩阵形式最小二乘
- [ ] 掌握多重共线性的诊断与岭回归动机
- [ ] 能够评估模型 $R^2$、Adjusted $R^2$

### 任务描述
- [ ] 创建 MultivariateLinearRegression 类，实现正规方程与梯度下降
- [ ] 编写单元测试类 HousePriceMultiTest，比较加入"楼层、房龄"后效果

### 场景
- [ ] 同一城市房价数据，特征包括面积、楼层、建成年份

### 产出要求
- [ ] 核心算法类：MultivariateLinearRegression.java
- [ ] 单元测试类：HousePriceMultiTest.java
- [ ] 测试数据集：house_price_multi.csv

---

## 任务3：最小二乘法实现

### 学习目标
- [ ] 复习最小二乘法与误差平方和最小化思想
- [ ] 掌握拟合优度、残差正态性检验
- [ ] 理解最小二乘与最大似然的等价

### 任务描述
- [ ] 创建 OrdinaryLeastSquares 类，提供一元/多元 OLS 通用实现
- [ ] 编写单元测试类 BookSalesForecastTest，预测连锁书店每日销量

### 场景
- [ ] 连锁书店 30 天销量与促销天数、天气指数等特征

### 产出要求
- [ ] 核心算法类：OrdinaryLeastSquares.java
- [ ] 单元测试类：BookSalesForecastTest.java
- [ ] 测试数据集：book_sales.csv

---

## 任务4：LASSO 回归实现

### 学习目标
- [ ] 理解 L1 正则化产生稀疏解的机制
- [ ] 掌握坐标下降法求解 LASSO
- [ ] 能够用交叉验证选择正则系数

### 任务描述
- [ ] 创建 LassoRegression 类，实现坐标下降与交叉验证
- [ ] 编写单元测试类 GeneExpressionSelectionTest，比较特征数量与预测准确率

### 场景
- [ ] 癌症基因表达数据，高维特征下选取关键基因

### 产出要求
- [ ] 核心算法类：LassoRegression.java
- [ ] 单元测试类：GeneExpressionSelectionTest.java
- [ ] 测试数据集：gene_expression.csv

---

## 任务5：岭回归实现

### 学习目标
- [ ] 理解 L2 正则化抑制多重共线性
- [ ] 掌握闭式解加正则项推导
- [ ] 比较岭回归与线性回归在高方差场景下的表现

### 任务描述
- [ ] 创建 RidgeRegression 类，实现闭式解与交叉验证
- [ ] 编写单元测试类 HousePriceRidgeTest，对比 RMSE 与系数稳定性

### 场景
- [ ] 房价多特征数据（含高度相关特征：面积与周长）

### 产出要求
- [ ] 核心算法类：RidgeRegression.java
- [ ] 单元测试类：HousePriceRidgeTest.java
- [ ] 测试数据集：house_price_ridge.csv

---

## 任务6：弹性网络回归实现

### 学习目标
- [ ] 理解弹性网络结合 L1/L2 优势
- [ ] 掌握调参网格搜索选择 alpha, l1_ratio
- [ ] 分析稀疏度-方差权衡

### 任务描述
- [ ] 创建 ElasticNetRegression 类，实现坐标下降 + 交叉验证
- [ ] 编写单元测试类 AdClickThroughRateTest，预测广告点击率

### 场景
- [ ] 高维广告 One-Hot 特征与点击标签

### 产出要求
- [ ] 核心算法类：ElasticNetRegression.java
- [ ] 单元测试类：AdClickThroughRateTest.java
- [ ] 测试数据集：ad_ctr.csv

---

## 任务7：PCA 回归一体化实现

### 学习目标
- [ ] 掌握协方差矩阵、特征值分解
- [ ] 理解降维后回归的 Bias/Variance 变化
- [ ] 熟悉主成分个数的选择策略

### 任务描述
- [ ] 创建 PCARegressor 类，先 PCA 降到 k 维再线性回归
- [ ] 编写单元测试类 MNISTStrokeWidthTest，预测数字粗笔画宽度

### 场景
- [ ] MNIST 图片 → 像素灰度 → PCA → 预测粗细度量

### 产出要求
- [ ] 核心算法类：PCARegressor.java
- [ ] 单元测试类：MNISTStrokeWidthTest.java
- [ ] 测试数据集：mnist_stroke.csv

---

## 任务8：流形学习回归实现

### 学习目标
- [ ] 理解非线性降维（Isomap / LLE）
- [ ] 掌握局部保持映射与全局几何保持
- [ ] 评估降维后回归的误差

### 任务描述
- [ ] 创建 ManifoldLearningRegression 类，支持 Isomap、LLE 两种方法
- [ ] 编写单元测试类 SwissRollRegressionTest，重建瑞士卷生成参数

### 场景
- [ ] 生成 3D 瑞士卷数据，目标回归变量为生成角度 t

### 产出要求
- [ ] 核心算法类：ManifoldLearningRegression.java
- [ ] 单元测试类：SwissRollRegressionTest.java
- [ ] 测试数据集：swiss_roll.csv

---

## 任务9：朴素贝叶斯实现

### 学习目标
- [ ] 理解朴素贝叶斯分类器的生成模型
- [ ] 掌握条件概率计算与预测公式
- [ ] 学习处理离散与连续特征（高斯 NB）

### 任务描述
- [ ] 创建 NaiveBayesClassifier 类，实现多项式与高斯朴素贝叶斯
- [ ] 编写单元测试类 EmailSpamDetectionTest，比较准确率与召回率

### 场景
- [ ] Enron 简化垃圾邮件数据集，词频 + 词长度特征

### 产出要求
- [ ] 核心算法类：NaiveBayesClassifier.java
- [ ] 单元测试类：EmailSpamDetectionTest.java
- [ ] 测试数据集：enron_spam.csv

---

## 任务10：半朴素贝叶斯实现

### 学习目标
- [ ] 理解半朴素贝叶斯放宽独立性假设
- [ ] 掌握特征依赖树 (TAN) 构建
- [ ] 比较模型复杂度与性能提升

### 任务描述
- [ ] 创建 SemiNaiveBayesClassifier 类，实现 TAN 算法
- [ ] 编写单元测试类 SpamTANTest，对比朴素贝叶斯

### 场景
- [ ] 同 Enron 数据集，测试特征依赖带来的提升

### 产出要求
- [ ] 核心算法类：SemiNaiveBayesClassifier.java
- [ ] 单元测试类：SpamTANTest.java
- [ ] 测试数据集：enron_spam.csv

---

## 任务11：贝叶斯网络实现

### 学习目标
- [ ] 理解 DAG、条件概率表 (CPT)
- [ ] 掌握前向后向推理与变量消除
- [ ] 能够设计中小型 BN

### 任务描述
- [ ] 创建 BayesianNetwork 类，支持添加节点、CPT 和推理
- [ ] 编写单元测试类 RespiratoryDiagnosisTest，判断流感概率

### 场景
- [ ] 呼吸道症状-疾病网络（咳嗽, 发烧, 流感）

### 产出要求
- [ ] 核心算法类：BayesianNetwork.java
- [ ] 单元测试类：RespiratoryDiagnosisTest.java
- [ ] 测试数据集：respiratory_bn.json

---

## 任务12：高斯贝叶斯网络实现

### 学习目标
- [ ] 掌握连续型 BN 的线性-高斯假设
- [ ] 理解参数学习与结构学习流程

### 任务描述
- [ ] 创建 GaussianBayesianNetwork 类，实现线性高斯 CPD
- [ ] 编写单元测试类 SensorFaultDetectionTest，检测工业传感器异常

### 场景
- [ ] 多维连续传感器读数与故障状态

### 产出要求
- [ ] 核心算法类：GaussianBayesianNetwork.java
- [ ] 单元测试类：SensorFaultDetectionTest.java
- [ ] 测试数据集：sensor_fault.csv

---

## 任务13：马尔可夫随机场实现

### 学习目标
- [ ] 理解无向图模型与势函数
- [ ] 掌握对数势、吉布斯能量与 MAP
- [ ] 熟悉迭代条件模式 (ICM) 与信念传播

### 任务描述
- [ ] 创建 MarkovRandomFieldDenoiser 类，实现 Ising 去噪模型
- [ ] 编写单元测试类 BinaryImageDenoiseTest，比对噪声-去噪图片

### 场景
- [ ] 32×32 二值图片 + 盐胡椒噪声

### 产出要求
- [ ] 核心算法类：MarkovRandomFieldDenoiser.java
- [ ] 单元测试类：BinaryImageDenoiseTest.java
- [ ] 测试数据集：binary_noise.png

---

## 任务14：条件随机场实现

### 学习目标
- [ ] 了解 CRF 线性链结构与特征函数
- [ ] 掌握前向-后向与维特比解码
- [ ] 实现 L-BFGS 参数训练

### 任务描述
- [ ] 创建 ConditionalRandomFieldSegmenter 类，实现中文分词
- [ ] 编写单元测试类 ChineseSegmentationCRFTest，计算 F1 值

### 场景
- [ ] 人民日报分词标注语料的子集

### 产出要求
- [ ] 核心算法类：ConditionalRandomFieldSegmenter.java
- [ ] 单元测试类：ChineseSegmentationCRFTest.java
- [ ] 测试数据集：people_daily_seg.txt

---

## 任务15：高斯过程分类实现

### 学习目标
- [ ] 理解 GP 先验与核函数
- [ ] 掌握拉普拉斯近似二分类
- [ ] 能够可视化后验分布

### 任务描述
- [ ] 创建 GaussianProcessClassifier 类，支持 RBF 核
- [ ] 编写单元测试类 BreastCancerGPCTest，预测乳腺癌良恶性

### 场景
- [ ] Wisconsin 乳腺癌数据集

### 产出要求
- [ ] 核心算法类：GaussianProcessClassifier.java
- [ ] 单元测试类：BreastCancerGPCTest.java
- [ ] 测试数据集：breast_cancer.csv

---

## 任务16：逻辑回归实现

### 学习目标
- [ ] 掌握对数似然与梯度上升
- [ ] 理解 L1/L2 正则对过拟合的影响

### 任务描述
- [ ] 创建 LogisticRegressionClassifier 类，实现批量/随机梯度下降
- [ ] 编写单元测试类 TitanicSurvivalTest

### 场景
- [ ] 泰坦尼克号乘客生存数据集

### 产出要求
- [ ] 核心算法类：LogisticRegressionClassifier.java
- [ ] 单元测试类：TitanicSurvivalTest.java
- [ ] 测试数据集：titanic.csv

---

## 任务17：线性判别分析实现

### 学习目标
- [ ] 理解类内/类间散度矩阵
- [ ] 掌握特征投影方向求解

### 任务描述
- [ ] 创建 LinearDiscriminantAnalysis 类，实现两类/多类 LDA
- [ ] 编写单元测试类 IrisLDATest，绘制 2D 投影

### 场景
- [ ] 鸢尾花数据集三类可视化

### 产出要求
- [ ] 核心算法类：LinearDiscriminantAnalysis.java
- [ ] 单元测试类：IrisLDATest.java
- [ ] 测试数据集：iris.csv

---

## 任务18：硬间隔 SVM 实现

### 学习目标
- [ ] 理解最大间隔几何解释
- [ ] 掌握原问题拉格朗日对偶

### 任务描述
- [ ] 创建 SVMHardMargin 类，使用 liblinear 方式求解
- [ ] 编写单元测试类 MoonHardMarginTest，分类可分月亮数据

### 场景
- [ ] make_moons 人工数据集

### 产出要求
- [ ] 核心算法类：SVMHardMargin.java
- [ ] 单元测试类：MoonHardMarginTest.java
- [ ] 测试数据集：moon.csv

---

## 任务19：对偶软间隔 SVM 实现

### 学习目标
- [ ] 理解对偶问题与支持向量概念
- [ ] 掌握 SMO 解法流程

### 任务描述
- [ ] 创建 SVMDualForm 类，实现线性核软间隔 SMO
- [ ] 编写单元测试类 MoonSoftMarginTest

### 场景
- [ ] 同可分月亮数据，加入 5% 标签噪声

### 产出要求
- [ ] 核心算法类：SVMDualForm.java
- [ ] 单元测试类：MoonSoftMarginTest.java
- [ ] 测试数据集：moon_soft.csv

---

## 任务20：RBF 核 SVM 实现

### 学习目标
- [ ] 掌握径向基核函数与参数 $\gamma$
- [ ] 理解核技巧在高维映射

### 任务描述
- [ ] 创建 SVMRbfKernel 类，实现 SMO + RBF
- [ ] 编写单元测试类 Digit0vs1SVMTest，分类 MNIST 0 与 1

### 场景
- [ ] MNIST 子集数字 0 与 1

### 产出要求
- [ ] 核心算法类：SVMRbfKernel.java
- [ ] 单元测试类：Digit0vs1SVMTest.java
- [ ] 测试数据集：mnist01.csv

---

## 任务21：多项式核 SVM 实现

### 学习目标
- [ ] 掌握多项式核超参数 degree, coef0
- [ ] 比较多项式核与 RBF 的训练时间

### 任务描述
- [ ] 创建 SVMPolyKernel 类
- [ ] 编写单元测试类 SentimentPolarityTest，文本情感正负分类

### 场景
- [ ] IMDb 影评二分类 TF-IDF 特征

### 产出要求
- [ ] 核心算法类：SVMPolyKernel.java
- [ ] 单元测试类：SentimentPolarityTest.java
- [ ] 测试数据集：imdb_sentiment.csv

---

## 任务22：Sigmoid 核 SVM 实现

### 学习目标
- [ ] 了解 Sigmoid 核与神经网络关系
- [ ] 掌握核参数 kappa, delta 调优

### 任务描述
- [ ] 创建 SVMSigmoidKernel 类
- [ ] 编写单元测试类 MovieRatingSVMTest

### 场景
- [ ] 同 IMDb 数据，比较三种核函数效果

### 产出要求
- [ ] 核心算法类：SVMSigmoidKernel.java
- [ ] 单元测试类：MovieRatingSVMTest.java
- [ ] 测试数据集：imdb_sentiment.csv

---

## 任务23：ID3 决策树实现

### 学习目标
- [ ] 理解信息熵与信息增益
- [ ] 掌握离散特征划分

### 任务描述
- [ ] 创建 DecisionTreeID3 类，实现递归建树与预测
- [ ] 编写单元测试类 WeatherPlayGolfTest

### 场景
- [ ] 经典 Weather-Play 数据

### 产出要求
- [ ] 核心算法类：DecisionTreeID3.java
- [ ] 单元测试类：WeatherPlayGolfTest.java
- [ ] 测试数据集：weather.csv

---

## 任务24：C4.5 决策树实现

### 学习目标
- [ ] 理解增益率与连续特征处理
- [ ] 掌握剪枝策略减少过拟合

### 任务描述
- [ ] 创建 DecisionTreeC45 类，支持连续阈值切分、预剪枝
- [ ] 编写单元测试类 AdultIncomeTreeTest

### 场景
- [ ] UCI Adult 收入数据，预测 >50K

### 产出要求
- [ ] 核心算法类：DecisionTreeC45.java
- [ ] 单元测试类：AdultIncomeTreeTest.java
- [ ] 测试数据集：adult_income.csv

---

## 任务25：CART 决策树实现

### 学习目标
- [ ] 理解基尼指数与二叉划分
- [ ] 掌握代价复杂度剪枝

### 任务描述
- [ ] 创建 DecisionTreeCART 类，可生成分类回归树
- [ ] 编写单元测试类 ChessboardColorTest，预测棋盘黑白格

### 场景
- [ ] 2D 坐标 → 黑 / 白 格标签

### 产出要求
- [ ] 核心算法类：DecisionTreeCART.java
- [ ] 单元测试类：ChessboardColorTest.java
- [ ] 测试数据集：chessboard.csv

---

## 任务26：AdaBoost 实现

### 学习目标
- [ ] 理解加权数据集与弱分类器组合
- [ ] 掌握指数损失与权重更新

### 任务描述
- [ ] 创建 AdaBoostClassifier 类，弱学习器用决策树桩
- [ ] 编写单元测试类 FaceDetectionAdaBoostTest

### 场景
- [ ] 人脸 Haar 特征小数据集

### 产出要求
- [ ] 核心算法类：AdaBoostClassifier.java
- [ ] 单元测试类：FaceDetectionAdaBoostTest.java
- [ ] 测试数据集：face_haar.csv

---

## 任务27：XGBoost 实现

### 学习目标
- [ ] 理解梯度提升树的二阶近似
- [ ] 掌握正则项与 shrinkage

### 任务描述
- [ ] 创建 XGBoostClassifier 类，使用近似分裂
- [ ] 编写单元测试类 CreditDefaultXGBoostTest

### 场景
- [ ] 信用卡违约预测数据

### 产出要求
- [ ] 核心算法类：XGBoostClassifier.java
- [ ] 单元测试类：CreditDefaultXGBoostTest.java
- [ ] 测试数据集：credit_default.csv

---

## 任务28：GBDT 实现

### 学习目标
- [ ] 理解负梯度残差作为新目标
- [ ] 掌握学习率与终止条件

### 任务描述
- [ ] 创建 GBDTClassifier 类，实现交叉熵损失
- [ ] 编写单元测试类 EcommerceClickGBDTTest

### 场景
- [ ] 电商商品点击预测

### 产出要求
- [ ] 核心算法类：GBDTClassifier.java
- [ ] 单元测试类：EcommerceClickGBDTTest.java
- [ ] 测试数据集：ecommerce_click.csv

---

## 任务29：随机森林实现

### 学习目标
- [ ] 掌握 Bagging + 随机子特征
- [ ] 理解 OOB 误差估计

### 任务描述
- [ ] 创建 RandomForestClassifier 类，支持并行建树
- [ ] 编写单元测试类 GeneCancerRandomForestTest

### 场景
- [ ] 基因表达癌症分类

### 产出要求
- [ ] 核心算法类：RandomForestClassifier.java
- [ ] 单元测试类：GeneCancerRandomForestTest.java
- [ ] 测试数据集：gene_expression.csv

---

## 任务30：Bagging 集成实现

### 学习目标
- [ ] 理解自助采样与方差降低
- [ ] 学会封装通用 Bagging 框架

### 任务描述
- [ ] 创建 BaggingEnsemble 类，可注入任意弱学习器
- [ ] 编写单元测试类 VoiceGenderBaggingTest

### 场景
- [ ] 声纹 MFCC 特征 → 性别识别

### 产出要求
- [ ] 核心算法类：BaggingEnsemble.java
- [ ] 单元测试类：VoiceGenderBaggingTest.java
- [ ] 测试数据集：voice_gender.csv

---

## 任务31：Boosting 框架实现

### 学习目标
- [ ] 抽象梯度 Boosting 过程
- [ ] 理解损失函数一阶梯度

### 任务描述
- [ ] 创建 GenericBoostingFramework 类，支持插件式弱模型

### 场景
- [ ] 波士顿房价回归

### 产出要求
- [ ] 核心算法类：GenericBoostingFramework.java
- [ ] 单元测试类：HousePriceGBRTest.java
- [ ] 测试数据集：boston_housing.csv

---

## 任务32：K 最近邻实现

### 学习目标
- [ ] 理解 KNN 懒惰学习思想
- [ ] 掌握效率提升的 KD-Tree 实现

### 任务描述
- [ ] 创建 KNearestNeighbors 类，支持多距离度量

### 场景
- [ ] MNIST 10 类 KNN 识别

### 产出要求
- [ ] 核心算法类：KNearestNeighbors.java
- [ ] 单元测试类：DigitKNNTest.java
- [ ] 测试数据集：mnist_knn.csv

---

## 任务33：隐马尔可夫模型实现

### 学习目标
- [ ] 理解 HMM 生成模型 (π, A, B)
- [ ] 掌握前向-后向、维特比、Baum-Welch

### 任务描述
- [ ] 创建 HiddenMarkovModel 类，实现离散 HMM 训练与解码

### 场景
- [ ] 小规模英文词性标注语料

### 产出要求
- [ ] 核心算法类：HiddenMarkovModel.java
- [ ] 单元测试类：PosTaggingHMMTest.java
- [ ] 测试数据集：pos_tagging.txt

---

## 任务34：线性动态系统-卡尔曼滤波实现

### 学习目标
- [ ] 掌握 Kalman Filter 预测-校正循环
- [ ] 理解噪声协方差矩阵作用

### 任务描述
- [ ] 创建 KalmanFilter 类

### 场景
- [ ] 无人机 xy 位置 + GPS 噪声测量

### 产出要求
- [ ] 核心算法类：KalmanFilter.java
- [ ] 单元测试类：DronePositionKalmanTest.java
- [ ] 测试数据集：drone_gps.csv

---

## 任务35：非线性动态系统-粒子滤波实现

### 学习目标
- [ ] 理解重要性采样与重采样
- [ ] 掌握粒子权重归一化

### 任务描述
- [ ] 创建 ParticleFilter 类

### 场景
- [ ] 1D 机器人位姿估计

### 产出要求
- [ ] 核心算法类：ParticleFilter.java
- [ ] 单元测试类：Robot1DSLAMTest.java
- [ ] 测试数据集：robot_slam.csv

---

## 任务36：MCMC 推断实现

### 学习目标
- [ ] 理解 Metropolis-Hastings 与 Gibbs 采样
- [ ] 掌握收敛诊断

### 任务描述
- [ ] 创建 MCMCInference 类，支持 MH 与 Gibbs

### 场景
- [ ] 硬币正面概率后验采样

### 产出要求
- [ ] 核心算法类：MCMCInference.java
- [ ] 单元测试类：BetaBinomialSamplingTest.java
- [ ] 测试数据集：coin_flips.csv

---

## 任务37：变量消除算法实现

### 学习目标
- [ ] 理解消除顺序对复杂度的影响
- [ ] 掌握因子乘积与求和

### 任务描述
- [ ] 创建 VariableElimination 类

### 场景
- [ ] 伞/天气 BN 推断

### 产出要求
- [ ] 核心算法类：VariableElimination.java
- [ ] 单元测试类：UmbrellaWeatherInferenceTest.java
- [ ] 测试数据集：umbrella_bn.json

---

## 任务38：置信传播实现

### 学习目标
- [ ] 理解树形图精确 BP 与 Loopy BP
- [ ] 掌握消息更新公式

### 任务描述
- [ ] 创建 BeliefPropagation 类，支持同步/异步更新

### 场景
- [ ] 简易 10×10 立体匹配 MRF

### 产出要求
- [ ] 核心算法类：BeliefPropagation.java
- [ ] 单元测试类：StereoVisionBPTest.java
- [ ] 测试数据集：stereo_left_right.png

---

## 任务39：变分贝叶斯推断实现

### 学习目标
- [ ] 理解 ELBO 与变分分布
- [ ] 掌握坐标上升优化

### 任务描述
- [ ] 创建 VariationalBayesLDA 类，实现 LDA 文本主题模型

### 场景
- [ ] 小型新闻语料 (3k 文档)

### 产出要求
- [ ] 核心算法类：VariationalBayesLDA.java
- [ ] 单元测试类：NewsLDAVBTest.java
- [ ] 测试数据集：news_corpus.txt

---

## 任务40：EM 算法实现

### 学习目标
- [ ] 理解 E-步期望与 M-步最大化
- [ ] 掌握 GMM 参数收敛判定

### 任务描述
- [ ] 创建 GaussianMixtureEM 类，支持全协方差

### 场景
- [ ] 鸢尾花数据聚类与真实标签对比

### 产出要求
- [ ] 核心算法类：GaussianMixtureEM.java
- [ ] 单元测试类：IrisGMMEMTest.java
- [ ] 测试数据集：iris.csv

---

## 任务41：变分 EM 实现

### 学习目标
- [ ] 理解变分 EM 近似后验
- [ ] 掌握对高斯混合的贝叶斯推断

### 任务描述
- [ ] 创建 VariationalEMGMM 类

### 场景
- [ ] 同鸢尾花数据

### 产出要求
- [ ] 核心算法类：VariationalEMGMM.java
- [ ] 单元测试类：IrisVariationalGMMTest.java
- [ ] 测试数据集：iris.csv

---

## 任务42：维特比算法实现

### 学习目标
- [ ] 掌握最优状态序列动态规划
- [ ] 理解维特比与前向算法区别

### 任务描述
- [ ] 创建 ViterbiDecoder 类

### 场景
- [ ] 简化语音音素 HMM

### 产出要求
- [ ] 核心算法类：ViterbiDecoder.java
- [ ] 单元测试类：SpeechPhonemeViterbiTest.java
- [ ] 测试数据集：phoneme_hmm.json

---

## 任务43：遗传算法实现

### 学习目标
- [ ] 理解遗传编码、选择、交叉、变异操作
- [ ] 掌握精英保留策略

### 任务描述
- [ ] 创建 GeneticAlgorithmTSP 类，求 50 城 TSP

### 场景
- [ ] 欧氏坐标随机生成 50 个城市

### 产出要求
- [ ] 核心算法类：GeneticAlgorithmTSP.java
- [ ] 单元测试类：TSPGATest.java
- [ ] 测试数据集：tsp50.csv

---

## 任务44：粒子群优化实现

### 学习目标
- [ ] 理解 PSO 的社会/个体速度更新
- [ ] 掌握参数 w, c1, c2 调节

### 任务描述
- [ ] 创建 ParticleSwarmOptimization 类，求 Rastrigin 极小值

### 场景
- [ ] 2D Rastrigin 函数

### 产出要求
- [ ] 核心算法类：ParticleSwarmOptimization.java
- [ ] 单元测试类：RastriginPSOTest.java
- [ ] 测试数据集：rastrigin_param.json

---

## 任务45：蚁群算法实现

### 学习目标
- [ ] 理解信息素挥发与启发式函数
- [ ] 掌握 ACS 与 MMAS 区别

### 任务描述
- [ ] 创建 AntColonyOptimization 类，求网格图最短路径

### 场景
- [ ] 20×20 网格加障碍

### 产出要求
- [ ] 核心算法类：AntColonyOptimization.java
- [ ] 单元测试类：GridPathACOTest.java
- [ ] 测试数据集：grid_map.txt

---

## 任务46：模拟退火实现

### 学习目标
- [ ] 理解 Metropolis 接受准则
- [ ] 掌握温度调度策略

### 任务描述
- [ ] 创建 SimulatedAnnealingSolver 类，求 15 数码最短步

### 场景
- [ ] 经典 15 Puzzle 随机初始状态

### 产出要求
- [ ] 核心算法类：SimulatedAnnealingSolver.java
- [ ] 单元测试类：FifteenPuzzleSATest.java
- [ ] 测试数据集：puzzle15.txt

---

## 任务47：凝聚层次聚类-单连接实现

### 学习目标
- [ ] 理解 AGNES 单连接距离
- [ ] 掌握距离矩阵更新

### 任务描述
- [ ] 创建 AgglomerativeClusteringSingleLink 类，输出树状图

### 场景
- [ ] 社交网络好友相似度矩阵

### 产出要求
- [ ] 核心算法类：AgglomerativeClusteringSingleLink.java
- [ ] 单元测试类：SocialFriendClusterTest.java
- [ ] 测试数据集：social_friend.csv

---

## 任务48：凝聚层次聚类-全连接实现

### 学习目标
- [ ] 理解全连接 (Complete) 距离
- [ ] 比较聚类结果差异

### 任务描述
- [ ] 创建 AgglomerativeClusteringCompleteLink 类

### 场景
- [ ] 微博话题词向量 Word2Vec

### 产出要求
- [ ] 核心算法类：AgglomerativeClusteringCompleteLink.java
- [ ] 单元测试类：WeiboTopicClusterTest.java
- [ ] 测试数据集：weibo_topic.csv

---

## 任务49：凝聚层次聚类-均连接实现

### 学习目标
- [ ] 理解平均距离与方差方法
- [ ] 可视化聚类树切断高度

### 任务描述
- [ ] 创建 AgglomerativeClusteringAverageLink 类

### 场景
- [ ] 城市公交站经纬度

### 产出要求
- [ ] 核心算法类：AgglomerativeClusteringAverageLink.java
- [ ] 单元测试类：BusStopGeoClusterTest.java
- [ ] 测试数据集：bus_stop.csv

---

## 任务50：分裂层次聚类实现

### 学习目标
- [ ] 理解 DIANA 自顶向下策略
- [ ] 掌握簇间最大距离计算

### 任务描述
- [ ] 创建 DivisiveClustering 类

### 场景
- [ ] 图书借阅相似度（Jaccard）

### 产出要求
- [ ] 核心算法类：DivisiveClustering.java
- [ ] 单元测试类：LibraryBorrowClusterTest.java
- [ ] 测试数据集：library_borrow.csv

---

## 任务51：K-均值聚类实现

### 学习目标
- [ ] 理解 SSE 目标函数
- [ ] 掌握 K-Means++ 初始化

### 任务描述
- [ ] 创建 KMeansClustering 类

### 场景
- [ ] 电商 RFM 客户分群

### 产出要求
- [ ] 核心算法类：KMeansClustering.java
- [ ] 单元测试类：CustomerRFMSegmentationTest.java
- [ ] 测试数据集：customer_rfm.csv

---

## 任务52：高斯混合聚类实现

### 学习目标
- [ ] 理解 GMM 聚类与软分配
- [ ] 掌握对数似然收敛

### 任务描述
- [ ] 创建 GaussianMixtureClustering 类

### 场景
- [ ] 语音 MFCC 特征聚类

### 产出要求
- [ ] 核心算法类：GaussianMixtureClustering.java
- [ ] 单元测试类：SpeechMFCCGMMTest.java
- [ ] 测试数据集：speech_mfcc.csv

---

## 任务53：DBSCAN 聚类实现

### 学习目标
- [ ] 理解密度可达与簇扩展
- [ ] 掌握 ε 和 minPts 调参

### 任务描述
- [ ] 创建 DBSCANClustering 类

### 场景
- [ ] 全球地震震中经纬度聚簇

### 产出要求
- [ ] 核心算法类：DBSCANClustering.java
- [ ] 单元测试类：EarthquakeEpicenterDBSCANTest.java
- [ ] 测试数据集：earthquake.csv

---

## 任务54：闵可夫斯基距离实现

### 学习目标
- [ ] 理解 p 范数与距离度量统一框架

### 任务描述
- [ ] 创建 MinkowskiDistance 类，实现 p 可配置

### 场景
- [ ] 随机 2D 点对距离计算

### 产出要求
- [ ] 核心算法类：MinkowskiDistance.java
- [ ] 单元测试类：MinkowskiDistanceTest.java
- [ ] 测试数据集：distance_points.csv

---

## 任务55：欧氏距离实现

### 学习目标
- [ ] 掌握最常用 L2 距离性质

### 任务描述
- [ ] 创建 EuclideanDistance 类

### 场景
- [ ] 同任务54 数据

### 产出要求
- [ ] 核心算法类：EuclideanDistance.java
- [ ] 单元测试类：EuclideanDistanceTest.java
- [ ] 测试数据集：distance_points.csv

---

## 任务56：曼哈顿距离实现

### 学习目标
- [ ] 理解 L1 距离与稀疏向量差异

### 任务描述
- [ ] 创建 ManhattanDistance 类

### 场景
- [ ] 同任务54 数据

### 产出要求
- [ ] 核心算法类：ManhattanDistance.java
- [ ] 单元测试类：ManhattanDistanceTest.java
- [ ] 测试数据集：distance_points.csv

---

## 任务57：切比雪夫距离实现

### 学习目标
- [ ] 掌握 L∞ 距离在棋盘路径问题

### 任务描述
- [ ] 创建 ChebyshevDistance 类

### 场景
- [ ] 同任务54 数据

### 产出要求
- [ ] 核心算法类：ChebyshevDistance.java
- [ ] 单元测试类：ChebyshevDistanceTest.java
- [ ] 测试数据集：distance_points.csv

---

## 任务58：马氏距离实现

### 学习目标
- [ ] 理解协方差归一化距离

### 任务描述
- [ ] 创建 MahalanobisDistance 类

### 场景
- [ ] 多元正态样本点距离计算

### 产出要求
- [ ] 核心算法类：MahalanobisDistance.java
- [ ] 单元测试类：MahalanobisDistanceTest.java
- [ ] 测试数据集：multivariate_normal.csv

---

## 任务59：余弦相似度实现

### 学习目标
- [ ] 掌握高维向量夹角度量

### 任务描述
- [ ] 创建 CosineSimilarity 类

### 场景
- [ ] 文本 TF-IDF 相似度

### 产出要求
- [ ] 核心算法类：CosineSimilarity.java
- [ ] 单元测试类：CosineSimilarityTest.java
- [ ] 测试数据集：tfidf_vectors.csv

---

## 任务60：皮尔逊相关系数实现

### 学习目标
- [ ] 理解线性相关与中心化

### 任务描述
- [ ] 创建 PearsonCorrelation 类

### 场景
- [ ] 股票收益率相关性

### 产出要求
- [ ] 核心算法类：PearsonCorrelation.java
- [ ] 单元测试类：PearsonCorrelationTest.java
- [ ] 测试数据集：stock_returns.csv

---

## 任务61：汉明距离实现

### 学习目标
- [ ] 理解二进制串差异计数

### 任务描述
- [ ] 创建 HammingDistance 类

### 场景
- [ ] DNA 序列编码比较

### 产出要求
- [ ] 核心算法类：HammingDistance.java
- [ ] 单元测试类：HammingDistanceTest.java
- [ ] 测试数据集：dna_sequences.csv

---

## 任务62：杰卡德相似系数实现

### 学习目标
- [ ] 理解集合交并比

### 任务描述
- [ ] 创建 JaccardSimilarity 类

### 场景
- [ ] 购物篮商品集合相似度

### 产出要求
- [ ] 核心算法类：JaccardSimilarity.java
- [ ] 单元测试类：JaccardSimilarityTest.java
- [ ] 测试数据集：market_basket.csv

---

## 任务63：编辑距离实现

### 学习目标
- [ ] 掌握动态规划 Levenshtein 算法

### 任务描述
- [ ] 创建 EditDistance 类

### 场景
- [ ] 拼写纠错单词对

### 产出要求
- [ ] 核心算法类：EditDistance.java
- [ ] 单元测试类：EditDistanceTest.java
- [ ] 测试数据集：word_pairs.csv

---

## 任务64：动态时间规整 (DTW) 距离实现

### 学习目标
- [ ] 理解时间序列对齐

### 任务描述
- [ ] 创建 DTWDistance 类

### 场景
- [ ] 加速度计手势模板匹配

### 产出要求
- [ ] 核心算法类：DTWDistance.java
- [ ] 单元测试类：DTWDistanceTest.java
- [ ] 测试数据集：gesture_acc.csv

---

## 任务65：KL 散度实现

### 学习目标
- [ ] 理解非对称概率分布距离

### 任务描述
- [ ] 创建 KLDivergence 类

### 场景
- [ ] 语言模型分布差异

### 产出要求
- [ ] 核心算法类：KLDivergence.java
- [ ] 单元测试类：KLDivergenceTest.java
- [ ] 测试数据集：language_model.csv

---

## 任务66：PCA 可视化实现

### 学习目标
- [ ] 熟悉特征值排序与方差解释率

### 任务描述
- [ ] 创建 PCAVisualizer 类，绘制 2D 散点

### 场景
- [ ] Fashion-MNIST 十类图片降至 2D

### 产出要求
- [ ] 核心算法类：PCAVisualizer.java
- [ ] 单元测试类：FashionMNIStPCAPlotTest.java
- [ ] 测试数据集：fashion_mnist.csv

---

## 任务67：包裹法特征选择实现

### 学习目标
- [ ] 理解 Wrapper + 外部模型

### 任务描述
- [ ] 创建 WrapperFeatureSelector 类，迭代式前向选择

### 场景
- [ ] KNN 回测选股指标

### 产出要求
- [ ] 核心算法类：WrapperFeatureSelector.java
- [ ] 单元测试类：StockIndicatorWrapperTest.java
- [ ] 测试数据集：stock_indicator.csv

---

## 任务68：过滤法特征选择实现

### 学习目标
- [ ] 掌握卡方检验/互信息

### 任务描述
- [ ] 创建 FilterFeatureSelector 类

### 场景
- [ ] 新闻分类词袋选词

### 产出要求
- [ ] 核心算法类：FilterFeatureSelector.java
- [ ] 单元测试类：TextChiSquareFilterTest.java
- [ ] 测试数据集：news_chi.csv

---

## 任务69：嵌入法特征选择实现

### 学习目标
- [ ] 理解 L1 正则内嵌稀疏选择

### 任务描述
- [ ] 创建 EmbeddedFeatureSelector 类，基于 L1 LogReg

### 场景
- [ ] 文本情感向量特征

### 产出要求
- [ ] 核心算法类：EmbeddedFeatureSelector.java
- [ ] 单元测试类：EmbeddedSelectionTest.java
- [ ] 测试数据集：imdb_sentiment.csv

---

## 任务70：LLE 流形学习实现

### 学习目标
- [ ] 掌握保持局部线性关系

### 任务描述
- [ ] 创建 LLEManifold 类

### 场景
- [ ] 瑞士卷降至 2D 可视化

### 产出要求
- [ ] 核心算法类：LLEManifold.java
- [ ] 单元测试类：SwissRollLLETest.java
- [ ] 测试数据集：swiss_roll.csv

---

## 任务71：t-SNE 可视化实现

### 学习目标
- [ ] 理解高维邻域概率分布
- [ ] 掌握过度拟合抖动处理

### 任务描述
- [ ] 创建 TSNEVisualizer 类

### 场景
- [ ] CIFAR-10 图片嵌入 2D

### 产出要求
- [ ] 核心算法类：TSNEVisualizer.java
- [ ] 单元测试类：CIFAR10TSNETest.java
- [ ] 测试数据集：cifar_embeddings.csv

---

## 任务72：正则化对比实验

### 学习目标
- [ ] 比较 L1/L2/ElasticNet 对过拟合

### 任务描述
- [ ] 创建 RegularizationComparison 类，输出学习曲线

### 场景
- [ ] 波士顿房价回归

### 产出要求
- [ ] 核心算法类：RegularizationComparison.java
- [ ] 单元测试类：RegularizationHousingTest.java
- [ ] 测试数据集：boston_housing.csv

---

## 任务73：统一降维接口实现

### 学习目标
- [ ] 设计 DimensionalityReducer 抽象类

### 任务描述
- [ ] 创建 DimensionalityReducer 接口与工厂模式

### 场景
- [ ] 调用不同算法统一 API

### 产出要求
- [ ] 核心算法类：DimensionalityReducer.java
- [ ] 单元测试类：ReducerFactoryTest.java
- [ ] 测试数据集：无

---

## 任务74：随机近似推断接口

### 学习目标
- [ ] 统一封装 MCMC / 重要性采样调用

### 任务描述
- [ ] 创建 StochasticApproximateInference 接口

### 场景
- [ ] β-Binomial 后验求均值

### 产出要求
- [ ] 核心算法类：StochasticApproximateInference.java
- [ ] 单元测试类：StochasticInferenceTest.java
- [ ] 测试数据集：coin_flips.csv

---

## 任务75：精确推断接口

### 学习目标
- [ ] 抽象 VariableElimination、JunctionTree 引擎

### 任务描述
- [ ] 创建 ExactInferenceEngine 接口，实现 VE

### 场景
- [ ] 伞-天气网络推断

### 产出要求
- [ ] 核心算法类：ExactInferenceEngine.java
- [ ] 单元测试类：ExactInferenceUmbrellaTest.java
- [ ] 测试数据集：umbrella_bn.json

---

## 任务76：确定近似推断接口

### 学习目标
- [ ] 统一 VB 与 Loopy BP

### 任务描述
- [ ] 创建 DeterministicApproximateInference 接口

### 场景
- [ ] LDA VB 与 Loopy BP 对比

### 产出要求
- [ ] 核心算法类：DeterministicApproximateInference.java
- [ ] 单元测试类：DeterministicInferenceTest.java
- [ ] 测试数据集：news_corpus.txt

---

## 任务77：马尔可夫链文本生成

### 学习目标
- [ ] 理解高阶 Markov 过程

### 任务描述
- [ ] 创建 MarkovChain 类，支持 n-gram

### 场景
- [ ] 小说片段生成下一词

### 产出要求
- [ ] 核心算法类：MarkovChain.java
- [ ] 单元测试类：TextMarkovGenerateTest.java
- [ ] 测试数据集：novel.txt

---

## 任务78：卡尔曼 vs. 粒子滤波对比

### 学习目标
- [ ] 比较线性/非线性滤波性能

### 任务描述
- [ ] 创建 FilterComparison 类，输出 RMSE 曲线

### 场景
- [ ] 同无人机轨迹数据，两种滤波器

### 产出要求
- [ ] 核心算法类：FilterComparison.java
- [ ] 单元测试类：DroneFilterCompareTest.java
- [ ] 测试数据集：drone_gps.csv

---

## 任务79：推断算法基准

### 学习目标
- [ ] 构建统一 Benchmark 工具

### 任务描述
- [ ] 创建 InferenceBenchmark 类，度量运行时间与 KL 散度

### 场景
- [ ] 对比任务 36/39/40 三种算法

### 产出要求
- [ ] 核心算法类：InferenceBenchmark.java
- [ ] 单元测试类：InferenceBenchmarkTest.java
- [ ] 测试数据集：benchmark_config.json
