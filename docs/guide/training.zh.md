# 训练

训练流程使用经授权的长时记录与标注，为各尺度构建活动分类器和归一化资产。

## 准备

请确认：

1. 已通过 `python -m pip install -e .` 安装软件包；
2. 经授权的训练流位于 `data/signals/train/`；
3. 标签位于 `data/annotations/train_annotations.csv`；
4. 选定的 PyTorch 安装可以访问目标 CUDA 设备。

首先运行公开健康检查：

~~~bash
python tests/smoke_test.py
~~~

## 命令

=== "顺序训练"

    ~~~bash
    python train.py
    ~~~

    在单个进程中训练已配置的工作流。

=== "并行训练"

    ~~~bash
    python train_parallel.py
    ~~~

    在资源充足的机器上协调相互独立的尺度／随机种子任务。

=== "Shell 封装"

    ~~~bash
    bash run_training.sh
    ~~~

    适合从源码检出目录运行长时间任务。

=== "tmux 封装"

    ~~~bash
    bash run_training_in_tmux.sh
    ~~~

    启动并记录持久终端训练会话。

## 实验

默认配置包括：

| 维度 | 取值 |
| --- | --- |
| 窗口长度 | 3 秒、5 秒、8 秒 |
| 步长 | 1 秒 |
| 集成随机种子 | 42、123、456 |
| 输入通道 | 6 个 ACC/GYRO 通道 |
| 分类器 | 背景 + 5 种前景活动 |

模型使用并行卷积核、双向 LSTM 和融合分类头。针对类别不均衡与嵌入分离实验，
还提供 Focal Loss 与 Triplet Loss。

## 参数

多项长时间运行控制可以在不修改源码的情况下调整：

~~~bash
export NUM_EPOCHS_STAGE2=100
export EARLY_STOPPING_PATIENCE=30
export MIN_EPOCHS_BEFORE_EARLY_STOP=40
python train.py
~~~

批量大小、学习率、窗口构建、数据增强和设备默认值位于
`imu_activity_pipeline.config`。请将所有非默认值与实验输出一并记录。

## 输出

一次完整运行可以写入：

~~~text
saved_models/
├── combined_model_3s_seed42.pth
├── combined_model_5s_seed123.pth
├── combined_model_8s_seed123.pth
├── combined_model_{3s,5s,8s}_seed{42,123,456}.pth
├── norm_params_3s.pkl
├── norm_params_5s.pkl
├── norm_params_8s.pkl
├── ensemble_config.json
├── logs/
└── plots/
~~~

本地生成的检查点、日志与图表默认被忽略，除非有意将其整理为可复现资产。
请始终将每个检查点与创建它时使用的归一化参数和配置配对保存。

## 复现

训练与校准完成后运行：

~~~bash
bash run_reproducibility_experiments.sh
~~~

该封装脚本协调已保存模型评估、内部稳健性检查、后处理策略对比、信号质量分析、
时间线图、外部队列压力测试与汇总图生成。

!!! warning

    此封装脚本不能替代资产准备。它需要[数据与模型资产](../reference/assets.md)中
    说明的本地文件，不会下载私有或公开数据集。
