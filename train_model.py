import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import copy
import json
import warnings
import kagglehub # Kaggle Hubをインポート

# PyTorch 2.6のweights_only=Trueに関する警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module='torch.serialization')

# Download latest version from Kaggle Hub
# データセットがダウンロードされるローカルパスが 'path' 変数に格納されます。
print("INFO: Kaggle Hubからデータセットをダウンロード中...")
try:
    path = kagglehub.dataset_download("misrakahmed/vegetable-image-dataset")
    print(f"INFO: データセットが {path} にダウンロードされました。")
except Exception as e:
    print(f"ERROR: Kaggle Hubからのデータセットダウンロードに失敗しました: {e}")
    print("INFO: データセットのパスを手動で設定するか、Kaggle APIキーを確認してください。")
    # ダウンロード失敗時のフォールバックパス（手動で設定する場合）
    # このパスは、ダウンロードされたデータセットのルートディレクトリを指すように調整してください。
    # 例: path = r"C:\Users\cyber\.cache\kagglehub\datasets\misrakahmed\vegetable-image-dataset\versions\1"
    path = r"C:\Users\cyber\.cache\kagglehub\datasets\misrakahmed\vegetable-image-dataset\versions\1" # フォールバックパスも修正


# 1. データセットのパスを設定
# ★★修正点★★: DATA_DIRをダウンロードされたデータセットのルートディレクトリに設定
# その後、'Vegetable Images'、'train'/'test' サブディレクトリにアクセスします。
DATA_ROOT_DIR = path # Kaggle Hubからダウンロードされたルートパス
MODEL_SAVE_PATH = 'models/grape_classifier.pt' # 学習済みモデルの保存パス
CLASS_NAMES_SAVE_PATH = 'models/class_names.json' # クラス名リストの保存パス

# GPUが利用可能か確認し、デバイスを設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"INFO: 使用デバイス: {device}")

# 2. データの前処理とデータ拡張を定義
data_transforms = {
    'train': transforms.Compose([ # キーを 'train' に変更
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([ # キーを 'test' に変更
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. データセットとデータローダーの作成
num_workers = 4 if device.type == 'cuda' else 0

# ★★修正点★★: TRAIN_DATA_PATHとTEST_DATA_PATHの構築方法を変更
# DATA_ROOT_DIRの下の 'Vegetable Images' フォルダ、さらにその下の 'train'/'test' フォルダを指すようにします。
TRAIN_DATA_PATH = os.path.join(DATA_ROOT_DIR, 'Vegetable Images', 'train')
TEST_DATA_PATH = os.path.join(DATA_ROOT_DIR, 'Vegetable Images', 'test')

print(f"INFO: Trainingデータセットのパス: {TRAIN_DATA_PATH}")
print(f"INFO: Testデータセットのパス: {TEST_DATA_PATH}")

try:
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DATA_PATH, data_transforms['train']),
        'test': datasets.ImageFolder(TEST_DATA_PATH, data_transforms['test'])
    }
except Exception as e:
    print(f"ERROR: ImageFolderのロードに失敗しました。データセットのディレクトリ構造を確認してください: {e}")
    print(f"期待されるパス: {TRAIN_DATA_PATH} および {TEST_DATA_PATH}")
    print("データセットの実際の構造に合わせて DATA_ROOT_DIR または os.path.join() の引数を修正してください。")
    exit() # エラーが発生した場合はスクリプトを終了

dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                              shuffle=True if x == 'train' else False,
                              num_workers=num_workers)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print(f"INFO: 検出されたクラス名: {class_names}")
print(f"INFO: クラス数: {len(class_names)}")

# 4. モデルのロードと最終層の変更
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 5. モデルの学習関数
def train_model(model, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']: # キーを 'train' と 'test' に変更
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test': # キーを 'test' に変更
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f'INFO: 検証精度が改善しませんでした。改善なしエポック数: {epochs_no_improve}/{patience}')

        print()

        if epochs_no_improve >= patience:
            print(f'INFO: 早期停止: 検証精度が {patience} エポック改善しなかったため、訓練を停止します。')
            break

    time_elapsed = time.time() - since
    print(f'訓練完了: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最高の検証精度: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # モデルの保存先ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CLASS_NAMES_SAVE_PATH), exist_ok=True)

    # モデルを学習
    print("INFO: モデル訓練を開始します...")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50, patience=10)

    # 学習済みモデルを保存
    torch.save(model_ft, MODEL_SAVE_PATH)
    print(f"INFO: 学習済みモデルが {MODEL_SAVE_PATH} に保存されました。")

    # クラス名を保存しておくと、FastAPI側で結果を解釈する際に便利です
    with open(CLASS_NAMES_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    print(f"INFO: クラス名が {CLASS_NAMES_SAVE_PATH} に保存されました。")
