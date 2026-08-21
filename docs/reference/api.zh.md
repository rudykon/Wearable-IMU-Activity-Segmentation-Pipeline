# Python API

软件包刻意保持小巧且面向源码。导入前请以可编辑模式安装：

~~~bash
python -m pip install -e .
~~~

## 软件包映射

| 模块 | 主要职责 |
| --- | --- |
| `imu_activity_pipeline.config` | 路径、划分名称、通道、窗口、类别及训练／解码默认值 |
| `signal_file_reader` | 读取每用户制表符分隔的传感器流 |
| `sensor_data_processing` | 滤波、特征准备、窗口构建与标签 |
| `neural_network_models` | 损失函数与 PyTorch 检测器／分类器定义 |
| `train`、`train_parallel` | 顺序与并行训练流程 |
| `inference` | 模型加载、多尺度预测、解码与片段生成 |
| `evaluate` | 同类别片段匹配与指标计算 |
| `prediction_writer` | 将输出行写入 Excel |

## 版本

~~~python
import imu_activity_pipeline

print(imu_activity_pipeline.__version__)
~~~

当前软件包版本：

~~~text
0.1.0
~~~

## 读取信号文件

`DataReader` 读取目录中的每个 `.txt` 文件，并返回以文件名主干为键的字典。

~~~python
from imu_activity_pipeline.signal_file_reader import DataReader

reader = DataReader("data/signals/external_test")
sessions = reader.read_data()

for user_id, frame in sessions.items():
    print(user_id, frame.shape)
~~~

每个数据帧应包含[数据](../guide/data.md)页面中说明的规范时间戳与六个 IMU 通道。

## 运行端到端推理

~~~python
from imu_activity_pipeline.inference import run_inference

segments = run_inference(
    data_dir="data/signals/internal_eval",
    output_file="predictions_internal_eval.xlsx",
)
~~~

返回的片段行与工作簿使用：

~~~text
user_id, category, start, end
~~~

## 写入片段记录

~~~python
from imu_activity_pipeline.prediction_writer import DataOutput

rows = [
    ["HNU00001", "跑步", 1760000000000, 1760000600000],
]

DataOutput(
    rows,
    output_file="predictions_external_test.xlsx",
).save_predictions()
~~~

## 实例化窗口分类器

~~~python
import torch

from imu_activity_pipeline.neural_network_models import CombinedModel

model = CombinedModel(
    input_channels=6,
    num_classes=6,
    window_size=300,
)

x = torch.randn(2, 300, 6)
logits = model(x)
print(logits.shape)  # torch.Size([2, 6])
~~~

实际模型将三种卷积核大小与双向 LSTM、融合分类头相结合。

## 配置

~~~python
from imu_activity_pipeline import config

print(config.SPLIT_NAMES)
print(config.WINDOW_CONFIGS)
print(config.ACTIVITIES)
~~~

导入 `config` 时会读取 `HLS_HAR_DATA_ROOT`、`HLS_HAR_MODEL_DIR` 等路径设置。
请在启动 Python 前设置环境变量。

## 兼容入口

仓库根目录的脚本在委托给软件包的同时保留源码检出命令：

~~~text
run_inference.py
train.py
train_parallel.py
train_single_model.py
evaluate.py
~~~

对于可复现自动化，当软件包模块提供 CLI 时，优先使用
`python -m imu_activity_pipeline.<module>`，并记录软件包版本／Git 提交。
