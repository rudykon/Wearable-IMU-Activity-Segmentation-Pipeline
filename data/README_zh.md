# 数据集获取说明

English version: [README.md](README.md)。

本 GitHub 仓库不直接分发数据集。本目录只保留运行代码所需的本地目录结构，获得授权的读者下载数据后，将文件放入对应目录即可复现实验。数据集访问说明统一维护在项目 GitHub 页面。

## 读者如何获取数据

在 PhysioNet 仓库正式发布前，研究用途数据需要 upon request 申请：

1. 打开本项目 GitHub 页面。
2. 在数据获取说明区域找到腾讯问卷调查申请链接。
3. 通过腾讯问卷调查提交研究用途、单位、联系方式和必要的数据使用信息。
4. 数据申请由负责数据管理的海南大学组织方审核。
5. 审核通过后，按邮件或表单反馈中的说明下载数据，并放入本目录。

在 PhysioNet 仓库正式发布后：

1. GitHub 页面上的申请表会关闭。
2. 读者改为直接通过 PhysioNet 仓库获取数据。
3. 本项目 GitHub 页面会继续维护最新的 PhysioNet 数据链接和引用信息。

## 本地目录结构

获得授权数据后，请按以下结构放置：

```text
data/
├── signals/
│   ├── train/
│   ├── internal_eval/
│   └── external_test/
├── annotations/
├── splits/
├── metadata/
└── public_external/
```

对应文件通常包括：

```text
data/signals/{train,internal_eval,external_test}/*.txt
data/annotations/*_annotations.csv
data/splits/*_users.txt
data/splits/split_manifest.csv
data/metadata/signal_manifest.csv
data/metadata/split_summary.csv
data/metadata/label_summary_by_split.csv
data/metadata/dataset_metadata.json
```

仓库 `.gitignore` 已配置为忽略这些本地数据文件，避免把需要申请的数据误提交到公开仓库。

## 信号文件格式

每个信号文件是 UTF-8 编码的制表符分隔 `.txt` 文件，第一行为传感器列名。时间戳为毫秒级 Unix 时间戳。默认活动分割模型使用 IMU 六通道：

```text
ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
```

正式发布的数据文件可能保留额外原始传感器列。

## 标注文件格式

`annotations/` 下的 CSV 文件字段如下：

```text
split,user_id,category,start,end
```

其中 `category` 取值为：

```text
羽毛球, 跳绳, 飞鸟, 跑步, 乒乓球
```

`start` 和 `end` 为毫秒级时间戳。
