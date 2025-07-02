import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import copy
import json # class_names.json の保存に必要

# 1. データセットのパスを設定
# あなたのローカルPC上のデータセットのルートパスを設定してください。
# C:/Users/student/flu/Fruit-Images-Dataset とのことですので、このパスに設定します。
DATA_DIR = "C:/Users/cyber/Downloads/Fruit-Images-Dataset-master/Fruit-Images-Dataset-master"
MODEL_SAVE_PATH = 'models/grape_classifier.pt' # 学習済みモデルの保存パス


# GPUが利用可能か確認 (ローカルPCにGPUがない場合は自動的にCPUが選択されます)
device = torch.device("cuda:0")
print(f"INFO: Using device: {device}")

# 2. データの前処理とデータ拡張を定義
# 画像のサイズを224x224にリサイズし、正規化を行います。
data_transforms = {
    'Training': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. データセットとデータローダーの作成
# num_workers: CPU環境では、データ読み込みの並列処理を無効にするため0を設定します。
#              これにより、マルチプロセスによるオーバーヘッドを避けることができます。
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['Training', 'Test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                             shuffle=True, num_workers=0) # CPU環境向けにnum_workersを0に固定
               for x in ['Training', 'Test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Test']}
class_names = image_datasets['Training'].classes
print(f"INFO: Detected class names: {class_names}")
print(f"INFO: Number of classes: {len(class_names)}")

# 4. モデルのロードと最終層の変更
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 5. モデルの学習関数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['Training', 'Test']:
            if phase == 'Training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'Training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'Test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # モデルの保存先ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # モデルを学習
    print("INFO: Starting model training...")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    # 学習済みモデルを保存
    torch.save(model_ft, MODEL_SAVE_PATH)
    print(f"INFO: Trained model saved to {MODEL_SAVE_PATH}")

    # クラス名を保存しておくと、FastAPI側で結果を解釈する際に便利です
    os.makedirs(os.path.dirname('models/class_names.json'), exist_ok=True)
    with open('models/class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    print("INFO: Class names saved to models/class_names.json")
