import pandas as pd
import torch
from dataset import ImageDataset
from datetime import datetime
from efficientnet_pytorch import EfficientNet
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transform import define_transform
from tqdm import tqdm
from typing import Any, Optional
from utils import pytorch_util

MODEL_NAME = 'efficientnet-b0'


def init_output(save_path: str) -> None:

    # 1. 檢查路徑是否存在，不存在就新增一個
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)


def train(model: Any, train_loader: DataLoader, val_loader: DataLoader,
          criterion: Any, optimizer: Any, save_dir: str, batch_size: int,
          epochs: int, device: torch.device) -> Any:

    # 1. 初始化資料
    best_loss = 0

    # 2. 開始訓練
    for epoch in range(epochs):

        # 初始化訓練變數
        train_loss = 0.0
        train_acc = 0.0
        train_tqdm = tqdm(train_loader)

        # 訓練
        model.train()
        for cnt, (data, label) in enumerate(train_tqdm, 1):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            _, predict_label = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (predict_label == label).sum()
            train_tqdm.set_description(f'Train epoch {epoch}')
            train_tqdm.set_postfix({
                'loss':
                float(train_loss) / cnt,
                'acc':
                float(train_acc) / (cnt * batch_size) * 100
            })

        # 初始化驗證變數
        val_loss = 0.0
        val_tqdm = tqdm(val_loader)
        val_acc = 0

        # 驗證
        model.eval()
        for cnt, (data, label) in enumerate(val_tqdm, 1):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            _, predict_label = torch.max(outputs, 1)
            val_acc += (predict_label == label).sum()

            loss = criterion(outputs, label)
            val_loss += loss.item()
            val_tqdm.set_description(f'Val epoch {epoch}')
            val_tqdm.set_postfix({
                'loss':
                float(val_loss) / cnt,
                'acc':
                float(val_acc) / (cnt * batch_size) * 100
            })

        if best_loss == 0:
            best_loss = float(val_loss) / len(val_loader)

        # 存檔
        if float(val_loss) / len(val_loader) < best_loss:

            best_loss = float(val_loss) / len(val_loader)
            checkpoint_path = Path(save_dir, 'best.pth')
            print(
                f'Checkpoint {checkpoint_path} saved!, val_loss={float(val_loss)/len(val_loader)}'
            )
            torch.save(model.state_dict(), str(checkpoint_path))

    # 3. 儲存最後一次的權重
    checkpoint_path = Path(save_dir, 'final.pth')
    torch.save(model.state_dict(), str(checkpoint_path))

    return model


def run(dataset_path: str,
        weight_path: Optional[str] = None,
        test_size: Optional[float] = 0.2,
        seed: Optional[int] = 100,
        batch_size: Optional[int] = 16,
        num_classes: Optional[int] = 2,
        lr: Optional[float] = 1e-3,
        epochs: Optional[int] = 10,
        device: Optional[str] = None):

    # 1. 讀取資料集路徑
    df_dataset = pd.read_csv(dataset_path)

    # 2. 分割訓練集與驗證集
    df_train, df_val = train_test_split(df_dataset,
                                        test_size=test_size,
                                        random_state=seed)

    train_transform, val_transform = define_transform()
    train_data = ImageDataset(df_train, train_transform)
    val_data = ImageDataset(df_val, val_transform)

    # 3. Dataloader
    # NOTE: num_workers參數在Windows需要設定為1，否則會有錯誤
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    # 4. 建立模型
    model = EfficientNet.from_name(MODEL_NAME)
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, num_classes)

    # 載入權重
    if weight_path is not None:
        print(weight_path)
        try:
            model.load_state_dict(torch.load(weight_path), strict=False)
        except Exception as e:
            print(
                'There is an error occurred while loading pretrained model weight, message shown as below:'
            )
            print(e)

    # 將模型般到指定的裝置上
    device = torch.device(
        device) if device is not None else pytorch_util.auto_device()
    model.to(device)

    # 5. 設定優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # 6. 訓練
    train(model, train_loader, val_loader, criterion, optimizer, save_dir,
          batch_size, epochs, device)


if __name__ == '__main__':

    # 1. 設定執行參數
    import argparse

    parser = argparse.ArgumentParser(description='EfficientNet training')
    parser.add_argument('-i',
                        '--dataset_path',
                        type=str,
                        required=True,
                        help='Path to csv dataset')
    parser.add_argument('-w',
                        '--weight_path',
                        type=str,
                        default=None,
                        help='Path to pretrained model weight')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=16,
                        help='Batch size, default is 16')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=1e-5,
                        help='Learning rate, default is 1e5')
    parser.add_argument('-ts',
                        '--test_size',
                        type=float,
                        default=0.2,
                        help='Test data size of entire data, default is 0.2')
    parser.add_argument('-sd',
                        '--seed',
                        type=int,
                        default=100,
                        help='dataset split random seed')
    parser.add_argument('-nc',
                        '--num_classes',
                        type=int,
                        required=True,
                        help='Number of classes')
    parser.add_argument('-ep',
                        '--epochs',
                        type=int,
                        default=10,
                        help='Number of classes')
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default='cuda',
        help='Specify device(cuda, cpu) for training, default is cuda')
    parser.add_argument('-o',
                        '--save_dir',
                        type=str,
                        default='outputs/',
                        help='Directory to output weight')

    args = parser.parse_args()

    # 2. 初始化存檔路徑
    current_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = Path(args.save_dir, current_str)
    init_output(save_dir)

    # 3. 開始訓練
    run(args.dataset_path, args.weight_path, args.test_size, args.seed,
        args.batch_size, args.num_classes, args.learning_rate, args.epochs)
