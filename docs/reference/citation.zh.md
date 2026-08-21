# 引用与许可

## 引用本软件

在项目专属存档 DOI 或论文引用发布之前，请引用带版本的仓库：

~~~bibtex
@software{kong_2026_wearable_imu_segmentation,
  author  = {Kong, Minghao},
  title   = {Wearable IMU Activity Segmentation Pipeline},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline},
  license = {Apache-2.0}
}
~~~

为保证报告可复现，请用训练或推理时使用的准确 Git 提交替换或补充 `version`。

## 建议的方法描述

> 我们使用 Wearable IMU Activity Segmentation Pipeline（版本 0.1.0，访问于所引
> Git 提交）处理六通道、100 Hz 加速度计与陀螺仪数据流，并结合对齐的 3、5、8 秒
> 分类器和时序片段解码。

请根据实际配置调整表述。不要声称使用了报告实验中并不存在的尺度、检查点、
解码器策略、Android 流程或数据集。

## 许可

仓库原创源代码以及分发的 Python 和 Android 模型资产采用
[Apache License 2.0](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/LICENSE)
许可。模型资产旁保留了各范围专用的许可副本。

Apache-2.0 不会改变：

- 参与者数据访问限制；
- 另行取得的数据集条款；
- 第三方依赖许可；
- 引用外部数据集与方法的要求。

## 项目链接

- [源码仓库](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline)
- [Issue 跟踪器](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/issues)
- [数据集访问说明](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/data/README.md)
- [Android 模型卡](https://github.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/blob/main/android_realtime_app/MODEL_CARD.md)
