import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import copy
import json # class_names.json の保存に必要
import warnings

# PyTorch 2.6のweights_only=Trueに関する警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module='torch.serialization')

# 1. データセットのパスを設定
# あなたのローカルPC上のデータセットのルートパスを設定してください。
# C:/Users/student/flu/Fruit-Images-Dataset とのことですので、このパスに設定します。
DATA_DIR = r"C:\Users\cyber\Downloads\Fruit-Images-Dataset-master\Fruit-Images-Dataset-master"
MODEL_SAVE_PATH = 'models/grape_classifier.pt' # 学習済みモデルの保存パス
CLASS_NAMES_SAVE_PATH = 'models/class_names.json' # クラス名リストの保存パス

# GPUが利用可能か確認 (ローカルPCにGPUがない場合は自動的にCPUが選択されます)
device = torch.device("cuda:0")
print(f"INFO: 使用デバイス: {device}")

# 2. データの前処理とデータ拡張を定義
# 訓練データにはより積極的なデータ拡張を適用し、モデルの汎化性能を高めます。
# 検証データには決定的な前処理のみを適用します。
data_transforms = {
    'Training': transforms.Compose([
        transforms.RandomResizedCrop(224), # ランダムにクロップし、224x224にリサイズ
        transforms.RandomHorizontalFlip(), # 50%の確率で水平方向に反転
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 色のランダムな変更
        transforms.RandomRotation(15), # ランダムに15度まで回転
        transforms.ToTensor(), # PIL ImageをPyTorchテンソルに変換 (0-1に正規化)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNetの統計情報で正規化
    ]),
    'Test': transforms.Compose([
        transforms.Resize(256), # まず256にリサイズ
        transforms.CenterCrop(224), # 中央を224x224にクロップ
        transforms.ToTensor(), # PIL ImageをPyTorchテンソルに変換 (0-1に正規化)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNetの統計情報で正規化
    ]),
}

# 3. データセットとデータローダーの作成
# num_workers: GPUを使用する場合は、データ読み込みを高速化するためにCPUコア数に合わせて設定します。
# CPU環境では、マルチプロセスによるオーバーヘッドを避けるため0を設定します。
num_workers = 4 if device.type == 'cuda' else 0 # GPUがあれば4、なければ0

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['Training', 'Test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                              shuffle=True if x == 'Training' else False, # 訓練データのみシャッフル
                              num_workers=num_workers)
               for x in ['Training', 'Test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Test']}
class_names = image_datasets['Training'].classes
print(f"INFO: 検出されたクラス名: {class_names}")
print(f"INFO: クラス数: {len(class_names)}")

# 4. モデルのロードと最終層の変更
# ResNet18モデルをImageNetで事前学習済みの重みでロードします。
# これにより、少ないデータで高い精度を達成しやすくなります（転移学習）。
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features # 最終全結合層の入力特徴量数
# 最終層を、データセットのクラス数に合わせて変更します。
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device) # モデルをGPUまたはCPUに移動

# 損失関数と最適化手法の定義
# AdamWオプティマイザは、SGDよりも一般的に収束が速く、性能が良い傾向があります。
criterion = nn.CrossEntropyLoss() # 分類問題のためのクロスエントロピー損失
optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.001) # AdamWオプティマイザを使用
# 学習率スケジューラ: 7エポックごとに学習率を0.1倍にします。
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 5. モデルの学習関数
def train_model(model, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    """
    モデルを訓練する関数。早期停止機能を含みます。

    Args:
        model (nn.Module): 訓練するPyTorchモデル。
        criterion (nn.Module): 損失関数。
        optimizer (torch.optim.Optimizer): オプティマイザ。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学習率スケジューラ。
        num_epochs (int): 訓練するエポックの最大数。
        patience (int): 検証精度が改善しない場合に訓練を停止するまでのエポック数。
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # 最も良いモデルの重みを保存
    best_acc = 0.0 # 最も良い検証精度
    epochs_no_improve = 0 # 改善が見られないエポック数

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 各エポックには訓練フェーズと検証フェーズがあります
        for phase in ['Training', 'Test']:
            if phase == 'Training':
                model.train() # モデルを訓練モードに設定
            else:
                model.eval() # モデルを評価モードに設定

            running_loss = 0.0
            running_corrects = 0

            # データをイテレート
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 勾配をゼロに初期化
                optimizer.zero_grad()

                # 順伝播
                # 訓練フェーズでは勾配を計算し、検証フェーズでは計算しない
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # 最も高い確率を持つクラスを取得
                    loss = criterion(outputs, labels) # 損失を計算

                    # 逆伝播 + 最適化 (訓練フェーズのみ)
                    if phase == 'Training':
                        loss.backward() # 勾配を計算
                        optimizer.step() # パラメータを更新

                # 統計情報
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 学習率の更新 (訓練フェーズのみ)
            if phase == 'Training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 検証フェーズで最も良いモデルを追跡し、早期停止をチェック
            if phase == 'Test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0 # 改善したのでカウンタをリセット
                else:
                    epochs_no_improve += 1 # 改善しなかったのでカウンタをインクリメント
                    print(f'INFO: 検証精度が改善しませんでした。改善なしエポック数: {epochs_no_improve}/{patience}')

        print() # エポック間の区切り

        # 早期停止の条件チェック
        if epochs_no_improve >= patience:
            print(f'INFO: 早期停止: 検証精度が {patience} エポック改善しなかったため、訓練を停止します。')
            break

    time_elapsed = time.time() - since
    print(f'訓練完了: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最高の検証精度: {best_acc:4f}')

    # 最も良いモデルの重みをロード
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # モデルの保存先ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CLASS_NAMES_SAVE_PATH), exist_ok=True)

    # モデルを学習
    print("INFO: モデル訓練を開始します...")
    # num_epochsを増やし、patienceを設定して早期停止を有効にします。
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50, patience=10)

    # 学習済みモデルを保存
    # モデルオブジェクト全体を保存することで、FastAPI側でのロードが容易になります。
    torch.save(model_ft, MODEL_SAVE_PATH)
    print(f"INFO: 学習済みモデルが {MODEL_SAVE_PATH} に保存されました。")

    # クラス名を保存しておくと、FastAPI側で結果を解釈する際に便利です
    # class_namesの順序は、ImageFolderがディレクトリ名からクラスを作成する際のデフォルトのソート順です。
    # FastAPI側のclass_labelsとこの順序が一致していることを確認してください。
    with open(CLASS_NAMES_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    print(f"INFO: クラス名が {CLASS_NAMES_SAVE_PATH} に保存されました。")

