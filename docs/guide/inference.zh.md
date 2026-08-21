# 推理

推理会将输入目录中的每个长时传感器文件转换为带时间戳的前景活动片段。

## 默认命令

~~~bash
python run_inference.py
~~~

默认情况下，封装脚本读取：

~~~text
data/signals/external_test/*.txt
saved_models/ensemble_config.json
saved_models/*.pth
saved_models/*.pkl
~~~

并写入：

~~~text
predictions_external_test.xlsx
~~~

## 显式路径

如需指定输入目录与输出工作簿，请调用软件包模块：

~~~bash
python -m imu_activity_pipeline.inference \
  --data_dir data/signals/internal_eval \
  --output predictions_internal_eval.xlsx
~~~

当数据和模型位于仓库外部时，可以使用环境变量：

~~~bash
export HLS_HAR_DATA_ROOT=/absolute/path/to/data
export HLS_HAR_MODEL_DIR=/absolute/path/to/saved_models
export HLS_HAR_INFERENCE_SPLIT=external_test
python run_inference.py
~~~

## 一次运行中发生的步骤

1. 加载选定尺度的检查点与配套归一化参数。
2. 读取每条制表符分隔的会话记录。
3. 过滤并归一化六个物理 IMU 通道。
4. 以共同的一秒步长构建 3、5、8 秒窗口。
5. 生成并对齐各尺度类别概率。
6. 使用配置的 LBSA 策略融合各尺度。
7. 平滑概率并运行 Viterbi 序列解码。
8. 细化边界、消解重叠并应用片段策略。
9. 将前景片段记录写入工作簿。

## 输出契约

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

更底层的输入输出接口请查看 [Python API](../reference/api.md)。

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
    候选。请先检查概率与策略诊断信息，再考虑放宽过滤条件。
