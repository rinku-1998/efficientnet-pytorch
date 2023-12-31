# efficientnet-pytorch
整理 [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) 的專案，加入以下訓練與資料增強 (Data Augmentation) 的功能。

## 安裝

1. 安裝 Pytorch，根據目標環境安裝 CPU 或 GPU 版本
2. 安裝其餘套件

```shell
# 正式環境
$ pip install -r requirements.txt  # pip
$ poetry install  # poetry

# 開發環境
$ pip install -r requirements-dev.txt  # pip
$ poetry install --dev  # poetry
```

## 使用方法

1.  準備照片與資料集清單

    - 照片：可以為 `jpg` 或 `png` 格式
    - dataset.csv（資料集清單）：使用 csv 格式逐行編寫，範例格式如下：

```csv
img_path,label
data/imgrs/19AA4F35-6F65-4629-961B-411F5FF73BB6_0.jpg,0
```

2. 執行 `python train.py` 就會開始訓練，結果預設會保存在 `outputs/` 資料夾

## 說明

- 訓練

```shell
$ python trian.py -i [DATASET_PATH] -nc [NUM_CLASSES] -ep [EPOCHS]
```

| 參數名稱                 | 型態    | 必填 | 預設值     | 說明               | 備註          |
| ------------------------ | ------- | ---- | ---------- | ------------------ | ------------- |
| `-i`, `--dataset_path`   | String  | Y    |            | 資料集 CSV 路徑    |               |
| `-w`, `--weight_path`    | String  | N    |            | 權重檔路徑         |               |
| `-bs`, `--batch_size`    | Integer | N    | 16         | Batch size         |               |
| `-lr`, `--learning_rate` | Float   | N    | 1e-5       | 學習率             |               |
| `-ts`, `--test_size`     | String  | N    | 16         | 測試集比例         |               |
| `-sd`, `--seed`          | Integer | N    | 100        | 測試集分割隨機種子 |               |
| `-nc`, `--num_classes`   | Integer | Y    |            | 分類數量           |               |
| `-ep`, `--epochs`        | Integer | N    | 10         | 迭代次數           |               |
| `-d`, `--device`         | String  | N    | cuda       | 使用的裝置         | `cuda`或`cpu` |
|                          |
| `-o`, `--save_dir`       | String  | N    | `outputs/` | 結果輸出資料夾     |               |
