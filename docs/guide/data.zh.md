# 数据

仓库发布代码、选定的可复现模型资产、目录占位文件与访问说明，**不发布**参与者
传感器记录。

!!! important "仅使用经授权的数据"

    将下载的记录保存在已被忽略的 `data/` 目录中。请勿提交、再分发这些数据，
    也不要将其上传到 Issue、Pull Request 或实验产物中。

## 访问方式

在计划中的 PhysioNet 发布完成之前，研究用途申请遵循仓库
[数据集访问说明](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/data/README.md)
中维护的表单与审核流程。PhysioNet 发布后，该文件仍是最新仓库链接与引用信息的
规范入口。

## 规范本地目录结构

~~~text
data/
├── signals/
│   ├── train/
│   │   └── HNUxxxxx.txt
│   ├── internal_eval/
│   │   └── HNUxxxxx.txt
│   └── external_test/
│       └── HNUxxxxx.txt
├── annotations/
│   ├── train_annotations.csv
│   ├── internal_eval_annotations.csv
│   ├── external_test_annotations.csv
│   └── all_annotations.csv
├── splits/
│   ├── train_users.txt
│   ├── internal_eval_users.txt
│   ├── external_test_users.txt
│   └── split_manifest.csv
├── metadata/
│   ├── signal_manifest.csv
│   ├── split_summary.csv
│   ├── label_summary_by_split.csv
│   └── dataset_metadata.json
└── public_external/
    ├── har70plus/
    ├── opportunity/
    ├── pamap2/
    └── wisdm_phone/
~~~

数据划分名称属于公开接口的一部分：

- `train` — 模型开发；
- `internal_eval` — 开发／校准评估；
- `external_test` — 最终评估或默认推理。

## 传感器流模式

每条记录均为 UTF-8 编码的制表符分隔文件。默认模型需要：

| 列 | 含义 |
| --- | --- |
| `ACC_TIME` | 用于输出边界的毫秒时间戳 |
| `ACC_X`、`ACC_Y`、`ACC_Z` | 三轴加速度 |
| `GYRO_X`、`GYRO_Y`、`GYRO_Z` | 三轴角速度 |

发布的文件可能保留 PPG 时间戳、PPG 通道或其他原始列。默认活动模型读取六个
ACC/GYRO 通道。

示例表头：

~~~text
ACC_TIME	ACC_X	ACC_Y	ACC_Z	GYRO_X	GYRO_Y	GYRO_Z
~~~

## 标注模式

CSV 标注文件使用：

~~~text
split,user_id,category,start,end
~~~

`start` 与 `end` 为毫秒时间戳。`category` 取以下值之一：

| 标签 | 英文 |
| --- | --- |
| 羽毛球 | Badminton |
| 跳绳 | Jump rope |
| 飞鸟 | Fly |
| 跑步 | Running |
| 乒乓球 | Table tennis |

## 重定向本地路径

可以通过环境变量替换规范路径：

| 变量 | 用途 |
| --- | --- |
| `HLS_HAR_DATA_ROOT` | 替换完整 `data/` 根目录 |
| `HLS_HAR_TRAIN_DATA_DIR` | 仅替换训练信号目录 |
| `HLS_HAR_INTERNAL_EVAL_DATA_DIR` | 替换内部评估信号目录 |
| `HLS_HAR_EXTERNAL_TEST_DATA_DIR` | 替换外部测试信号目录 |
| `HLS_HAR_*_ANNOTATIONS_FILE` | 替换某一划分的标注 CSV |
| `HLS_HAR_MODEL_DIR` | 替换 `saved_models/` |

示例：

~~~bash
export HLS_HAR_DATA_ROOT=/mnt/authorized/imu_dataset
export HLS_HAR_MODEL_DIR=/mnt/models/imu_activity
python run_inference.py
~~~

## 公开可移植性检查

`experiments/public_temporal_record_layer_checks/` 下的可选适配器可在用户另行下载的
公开数据集上检查片段记录层。这些脚本不会下载数据集，也不会取代各数据集自身的
许可与引用条款。
