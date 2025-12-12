# Leakage Detection for Complex Water Distribution Network

本项目面向**复杂供水管网（Water Distribution Network, WDN）的泄漏检测与识别问题**，构建了一套从数据生成、特征提取、模型修正到在线检测与算法验证的完整实验框架。整体流程基于 **WNTR** 水力仿真工具，结合动态水力特征校正与模糊推理方法，实现对管网泄漏的鲁棒检测。

---

## 目录结构

```text
.
├── Dataset Generator
├── Experiment
│   ├── dataset.py
│   ├── file
│   ├── model
│   ├── model_correction
│   ├── detection
│   └── FRLOMA
│       ├── experiment.py
│       └── experiment.ipynb

```

## 模块说明

### 1. Dataset Generator

`Dataset Generator` 是基于 **WNTR** 的数据生成工具，用于构建不同工况下的管网泄漏数据集。
 主要功能包括：

- 设置不同泄漏位置、泄漏强度与时间模式
- 运行水力仿真并采集压力、流量等关键水力变量
- 为后续检测算法提供标准化输入数据

生成的数据将被后续实验模块统一加载和处理。

------

### 2. Experiment

#### 2.1 `Experiment/dataset.py`

数据集加载与预处理脚本，主要功能包括：

- 读取 `Dataset Generator` 生成的原始仿真数据
- 提取压力、流量等水力特征
- 将处理后的特征数据统一以 **pickle** 格式保存到 `Experiment/file` 目录中，便于后续快速加载与复现实验

#### 2.2 `Experiment/model`

该目录包含全局公共函数与配置文件，例如：

- 实验参数设置
- 算法运行所需的统一配置接口

#### 2.3 `Experiment/model_correction`

修正模型目录，用于构建**不同 DMA 区域之间的水泵流量–压力关系模型**。

包含的主要文件如下：

- `PUMP_DMA_AB.pickle`
   模拟 **AB 区域泄漏**，用于计算 `PUMP_C_model` 的流量–压力模型
- `PUMP_DMA_C.pickle`
   模拟 **C 区域泄漏**，用于计算 `PUMP_AB_model` 的流量–压力模型
- `PUMP_AB_model.pickle`
   AB 区域水泵的 **流量–传感器压力模型**
- `PUMP_C_model.pickle`
   C 区域水泵的 **流量–传感器压力模型**

这些修正模型为后续动态水力特征校正（DHFC）提供基础支撑。

#### 2.4 `Experiment/detection`

检测模型目录，包含完整的泄漏检测功能模块，主要包括：

- **DHFC（Dynamic Hydraulic Feature Correction）**
   动态水力特征校正模块，用于消除工况变化对传感器信号的影响
- **SFLI（Sparse Fuzzy Leakage Identification）**
   基于模糊推理与稀疏特征的泄漏识别模块

该目录下集中实现了所有与泄漏检测相关的核心算法函数。

------

### 3. FRLOMA 模块

`Experiment/FRLOMA` 目录包含 **FRLOMA（Fuzzy Robust Leakage Online Monitoring Algorithm）** 的完整实现与实验分析代码。

#### 3.1 `FRLOMA/experiment.py`

FRLOMA 算法的实验启动脚本

- 执行 FRLOMA 在线泄漏检测流程
- 输出检测结果与关键性能指标

#### 3.2 `FRLOMA/experiment.ipynb`

基于 Jupyter Notebook 的实验分析脚本，用于：

- 可视化 FRLOMA 算法的检测结果
- 分析不同工况下的检测性能
- 对比不同参数或算法配置的实验表现
