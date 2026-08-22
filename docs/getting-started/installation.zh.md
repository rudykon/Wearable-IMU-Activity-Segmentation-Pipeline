# 安装

项目面向 **Python 3.12 或更高版本**，源码位于 `src/imu_activity_pipeline/`。
建议采用可编辑安装，使仓库根目录入口、实验脚本、Notebook 与直接 Python 导入
都解析到同一个软件包。

## 环境要求

- Python ≥ 3.12
- Conda，或 Python `venv` + pip
- Git
- 正常模型训练需要支持 CUDA 的环境
- 仅在构建 Android 应用时需要 JDK 17 + Android SDK

!!! note

    公开冒烟测试刻意设计为可在 CPU 上安全运行，并且只使用很小的临时文件。
    完整训练与推理依赖[数据与模型资产](../reference/assets.md)中说明的数据和模型文件。

## 推荐方式：Conda

~~~bash
git clone https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline.git
cd Wearable-IMU-Activity-Segmentation-Pipeline

conda env create -f environment.yml
conda activate imu-activity-pipeline
python -m pip install -e .
~~~

该环境固定了仓库使用的数值计算与机器学习软件栈，其中包括 PyTorch 2.5.1。

## 备选方式：pip

=== "Linux / macOS"

    ~~~bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ~~~

=== "Windows PowerShell"

    ~~~powershell
    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ~~~

如需针对特定 CUDA 版本安装 PyTorch，请使用
[PyTorch 官方安装器](https://pytorch.org/get-started/locally/)选择适配主机驱动的
wheel，然后安装项目的其余依赖。pip 依赖使用与 Hugging Face ZeroGPU 兼容的
PyTorch 2.8.0；复现用 Conda 环境仍保留原始 PyTorch 2.5.1 研究软件栈。

## 验证检出内容

~~~bash
python -c "import imu_activity_pipeline; print(imu_activity_pipeline.__version__)"
python tests/smoke_test.py
~~~

预期软件包版本：

~~~text
0.1.0
~~~

冒烟测试检查：

1. 软件包导入与规范路径；
2. 一个极小的临时制表符分隔信号流；
3. 标注解析；
4. 预测工作簿写入。

## 在本地预览本文档

文档依赖与研究环境相互隔离：

~~~bash
python -m pip install -r requirements-docs.txt
mkdocs serve
~~~

打开 `http://127.0.0.1:8000/`。严格生产构建命令为：

~~~bash
mkdocs build --strict
~~~

## 常见安装问题

??? question "无法导入软件包"

    在仓库根目录运行 `python -m pip install -e .`，并确认当前解释器与
    `python -m pip --version` 显示的解释器一致。

??? question "PyTorch 无法识别 GPU"

    分别检查驱动与已安装的 PyTorch wheel。项目要求固定了框架版本，
    但 CUDA wheel 的选择取决于主机环境。

??? question "冒烟测试通过，但推理无法启动"

    这通常表示代码安装正常，但缺少必要的本地信号文件或模型资产。
    请继续查看[快速开始](quickstart.md)和[资产映射](../reference/assets.md)。
