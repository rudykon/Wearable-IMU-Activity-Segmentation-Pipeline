# Dataset Access

Chinese version: [README_zh.md](README_zh.md).

The dataset is not distributed directly in this GitHub repository. This directory keeps only the expected local layout so that authorized users can place the data files in the correct locations after access is granted. Dataset access instructions are maintained on the project GitHub page.

## How to Get the Data

Before the PhysioNet repository is formally released, research-use data access is available upon request:

1. Open the GitHub page for this project.
2. Find the dataset access section and follow the linked Tencent Questionnaire request form.
3. Submit the required research-use information through the questionnaire.
4. The request is reviewed by the Hainan University organizer responsible for data management.
5. After approval, follow the provided instructions to download the dataset and place the files under this `data/` directory.

After the PhysioNet repository is released:

1. The GitHub access request form will be closed.
2. Readers should obtain the dataset directly from the PhysioNet repository.
3. The GitHub page will keep the current PhysioNet access link and citation information.

## Expected Local Layout

After authorized download, arrange the dataset as:

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

Expected files include:

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

The repository `.gitignore` intentionally excludes these local data assets, so authorized data files remain local and are not accidentally committed.

## Signal Format

Each signal file is a UTF-8 tab-separated `.txt` file. The first row is the sensor header. Timestamps are millisecond Unix timestamps. The default activity-segmentation model uses:

```text
ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z
```

Released files may preserve additional original sensor columns.

## Annotation Format

Annotation CSV files use:

```text
split,user_id,category,start,end
```

`category` is one of:

```text
羽毛球, 跳绳, 飞鸟, 跑步, 乒乓球
```

`start` and `end` are millisecond timestamps.
