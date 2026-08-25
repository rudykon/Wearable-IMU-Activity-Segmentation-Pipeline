# 复现

公开仓库支持两条不同路径：不需要数据的代码包检查，以及使用获授权参与者记录的完整实验。两者分开说明，避免让访问者误以为仓库已经包含数据。

## 公开验证

~~~bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
~~~

该冒烟测试可在 CPU 上运行，不需要参与者记录或模型检查点。

## 流程索引

| 步骤 | 文档 | 主要命令 |
| --- | --- | --- |
| 安装 | [环境与验证](getting-started/installation.md) | `python tests/smoke_test.py` |
| 放置获授权数据 | [数据集](guide/data.md)与[资产边界](reference/assets.md) | — |
| 训练后验模型 | [训练协议](guide/training.md) | `python train.py` |
| 生成活动记录 | [推理与 TRL](guide/inference.md) | `python run_inference.py` |
| 评估片段记录 | [片段级评估](guide/evaluation.md) | `python evaluate.py --split external_test` |
| 构建移动端原型 | [Android](deployment/android.md) | `./gradlew assembleDebug` |

Python 与 Android 权重托管于公开的 [Hugging Face 模型仓库](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline)。缺失资产会下载到约定目录，并依据 `model-assets.json` 校验。

## 完整实验入口

准备好本地数据与固定资产后运行：

~~~bash
bash run_reproducibility_experiments.sh
~~~

该脚本依次执行已保存模型的外部测试、内部鲁棒性与策略选择检查、代表性时间线生成、外部无标注队列压力测试和汇总图生成。输出保存在：

~~~text
experiments/results/
experiments/figures/
experiments/logs/
~~~

如需指定解释器：

~~~bash
PYTHON_BIN=/absolute/path/to/python bash run_reproducibility_experiments.sh
~~~

## 接口

- [快速开始](getting-started/quickstart.md)——最小获授权数据流程。
- [API](reference/api.md)——Python 包接口与输出格式。
- [模型与许可证](reference/assets.md)——文件名、哈希与分发边界。
- [引用](reference/citation.md)——按版本引用软件。

!!! warning "数据边界"

    公开仓库不存放参与者记录。复现或扩展研究时，必须保留文档规定的数据访问、隐私、划分与许可条件。
