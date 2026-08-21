# 使用指南

此页面保留仓库原有的 `docs/USAGE.md` 链接。扩展后的项目网站已将各工作流拆分为
独立指南：

- [安装](getting-started/installation.md)
- [快速开始](getting-started/quickstart.md)
- [训练](guide/training.md)
- [推理](guide/inference.md)
- [评估](guide/evaluation.md)
- [Python API](reference/api.md)
- [Android 部署](deployment/android.md)

## 最小公开验证

~~~bash
conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
python tests/smoke_test.py
~~~

## 默认经授权数据推理

~~~bash
python run_inference.py
~~~

默认数据划分为 `external_test`，输出为 `predictions_external_test.xlsx`。

## 实验封装脚本

~~~bash
bash run_reproducibility_experiments.sh
~~~

此命令需要[资产指南](reference/assets.md)中说明的本地数据与模型资产。
