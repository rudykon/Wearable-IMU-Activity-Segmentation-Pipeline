# 推理

把长时传感器文件转换成带时间戳的活动。

## 运行

~~~bash
python run_inference.py
~~~

读取：

~~~text
data/signals/external_test/*.txt
saved_models/ensemble_config.json
saved_models/*.pth
saved_models/*.pkl
~~~

写入：

~~~text
predictions_external_test.xlsx
~~~

## 路径

指定输入与输出：

~~~bash
python -m imu_activity_pipeline.inference \
  --data_dir data/signals/internal_eval \
  --output predictions_internal_eval.xlsx
~~~

外部数据与模型：

~~~bash
export HLS_HAR_DATA_ROOT=/absolute/path/to/data
export HLS_HAR_MODEL_DIR=/absolute/path/to/saved_models
export HLS_HAR_INFERENCE_SPLIT=external_test
python run_inference.py
~~~

## 步骤

1. 加载检查点和归一化参数。
2. 读取会话。
3. 滤波并归一化六个通道。
4. 构建 3、5、8 秒窗口。
5. 对齐概率。
6. 用 LBSA 融合尺度。
7. 用 Viterbi 解码序列。
8. 修正并过滤片段。
9. 写入工作簿。

## 输出

| 列 | 说明 |
| --- | --- |
| `user_id` | 文件名主干／会话标识符 |
| `category` | 五种前景活动标签之一 |
| `start` | 片段开始时间，单位为毫秒 |
| `end` | 片段结束时间，单位为毫秒 |

示例：

~~~text
HNU00001,跑步,1760000000000,1760000600000
~~~

## Python 接口

~~~python
from imu_activity_pipeline.inference import run_inference

segments = run_inference(
    data_dir="data/signals/internal_eval",
    output_file="predictions_internal_eval.xlsx",
)

print(f"generated {len(segments)} segments")
~~~

底层输入输出见 [Python API](../reference/api.md)。

## 故障排查

??? question "未找到输入文件"

    确认选定目录包含 `.txt` 文件，并检查当前数据划分或 `--data_dir` 是否指向该目录。

??? question "检查点可以加载，但张量形状不匹配"

    检查选定检查点是否与配置的窗口长度对应，并确认预期的六通道输入顺序未改变。

??? question "预测结果不稳定"

    在调节时序阈值前，先核对采样率、传感器佩戴位置、物理单位、滤波、各尺度
    归一化参数与集成配置。

??? question "工作簿为空"

    解码器可能将整条记录分类为背景，或通过持续时间／置信度策略移除了全部前景
    候选。请先检查概率图和过滤设置，再考虑放宽过滤条件。
