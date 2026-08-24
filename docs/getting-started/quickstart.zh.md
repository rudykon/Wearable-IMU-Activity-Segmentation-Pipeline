# 快速开始

本指南先验证公开代码可以正常运行，再说明如何用最少步骤将一组经授权的传感器流
转换为活动工作簿。

## 1. 安装

~~~bash
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
~~~

此步骤不需要私有记录或已训练检查点。

## 2. 目录

~~~text
data/
├── signals/
│   ├── train/
│   ├── internal_eval/
│   └── external_test/
├── annotations/
├── splits/
└── metadata/

saved_models/
├── ensemble_config.json
├── combined_model_3s_seed42.pth
├── combined_model_5s_seed123.pth
├── combined_model_8s_seed123.pth
├── norm_params_3s.pkl
├── norm_params_5s.pkl
└── norm_params_8s.pkl
~~~

每个传感器文件均采用 UTF-8 编码、制表符分隔，并至少包含：

~~~text
ACC_TIME  ACC_X  ACC_Y  ACC_Z  GYRO_X  GYRO_Y  GYRO_Z
~~~

默认模型使用六个物理通道。`ACC_TIME` 保存毫秒时间戳；发布的文件可以保留其他列。

## 3. 推理

将经授权的输入文件放入 `data/signals/external_test/`，然后运行：

~~~bash
python run_inference.py
~~~

兼容入口会调用已安装的软件包并写入：

~~~text
predictions_external_test.xlsx
~~~

每行预测均遵循：

~~~text
user_id, category, start, end
~~~

## 4. 路径

~~~bash
python -m imu_activity_pipeline.inference \
  --data_dir data/signals/internal_eval \
  --output predictions_internal_eval.xlsx
~~~

也可以在不移动仓库检出内容的情况下重定向规范路径：

~~~bash
export HLS_HAR_DATA_ROOT=/absolute/path/to/data
export HLS_HAR_MODEL_DIR=/absolute/path/to/saved_models
export HLS_HAR_INFERENCE_SPLIT=internal_eval
python run_inference.py
~~~

## 5. 评估

~~~bash
python evaluate.py \
  --split internal_eval \
  --predictions predictions_internal_eval.xlsx
~~~

评估器执行同类别、一对一的片段匹配，并报告 IoU > 0.5 时的精确率、召回率和 F1。

## 6. 复现

准备好所需的本地数据与固定资产后，运行：

~~~bash
bash run_reproducibility_experiments.sh
~~~

生成的材料保存在已被忽略的目录中：

~~~text
experiments/results/
experiments/figures/
experiments/logs/
~~~

如有需要，可以指定 Python 解释器：

~~~bash
PYTHON_BIN=/absolute/path/to/python bash run_reproducibility_experiments.sh
~~~

## 下一步

- 了解[端到端架构](../guide/pipeline.md)。
- 核对准确的[数据模式与访问边界](../guide/data.md)。
- 查看[推理与时序后处理](../guide/inference.md)。
- 构建 [Android 端侧演示](../deployment/android.md)。
