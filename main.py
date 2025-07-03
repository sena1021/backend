import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import os
from fastapi.middleware.cors import CORSMiddleware # CORSミドルウェアをインポート
import warnings

# PyTorch 2.6のweights_only=Trueに関する警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module='torch.serialization')

app = FastAPI()

# CORS設定を更新: すべてのlocalhostオリジンを許可するように変更
# 注意: allow_credentials=True の場合、allow_origins=["*"] は使用できません。
# 開発目的で全てのlocalhostからのアクセスを許可するために、allow_credentials を False に設定します。
# 本番環境では、具体的なオリジンを指定することを強く推奨します。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンからのアクセスを許可
    allow_credentials=False, # ★変更: allow_origins=["*"] と併用するため False に設定
    allow_methods=["*"],  # すべてのHTTPメソッドを許可 (GET, POSTなど)
    allow_headers=["*"],  # すべてのHTTPヘッダーを許可
)


# モデルのパス
# SyntaxErrorを避けるため、raw string (r"...") を使用してパスを定義します。
MODEL_PATH = r"C:\Users\student\back\backend\models\grape_classifier.pt"

# --- モデルのアーキテクチャ定義 ---
# ★★重要★★: ここに定義するGrapeClassifierクラスは、
# grape_classifier.pt が訓練されたときのモデルの正確なアーキテクチャと一致する必要があります。
# もし、ResNetなどのtorchvision.modelsの既存モデルを訓練して保存した場合、
# そのモデルを直接ロードする必要があります。
# 例:
# import torchvision.models as models
# class GrapeClassifier(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.model = models.resnet18(weights=None) # weights=Noneで事前学習済み重みをロードしない
#         # 最終層をカスタムのnum_classesに合わせる
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_ftrs, num_classes)
#
#     def forward(self, x):
#         return self.model(x)

# 現在のGrapeClassifierはシンプルなCNNの例です。
# 実際のモデルがこれと異なる場合、この定義を修正してください。
class GrapeClassifier(nn.Module):
    def __init__(self, num_classes=131): # クラス数を131に更新
        super().__init__()
        # ここに実際のモデルのレイヤーと構造を記述してください
        # 例:
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            # 以下のLinear層の入力サイズは、MaxPool2dの出力と入力画像サイズに依存します。
            # 例として224x224の画像で2回の2x2プーリング後、特徴マップが56x56になることを想定。
            # 実際のモデルの入力サイズとプーリング層の構成に合わせてこの値を調整してください。
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes) # 出力層のクラス数を更新
        )

    def forward(self, x):
        return self.features(x)

# モデルのロード (アプリケーション起動時に一度だけロードする)
# グローバル変数としてモデルをロードすることで、リクエストごとにロードするオーバーヘッドを避けます。
model = None
try:
    if os.path.exists(MODEL_PATH):
        # torch.loadが直接モデルインスタンスを返すことを想定し、
        # weights_only=False を指定してロードします。
        # モデルのソースを信頼できる場合にのみこのオプションを使用してください。
        # もし、grape_classifier.ptがtorch.save(model.state_dict(), ...)で保存された場合、
        # 上記のGrapeClassifierクラスのインスタンスを作成し、model.load_state_dict()を使用する必要があります。
        # 現在のエラーメッセージから、モデルインスタンス全体が保存されている可能性が高いです。
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        model.eval() # 推論モードに設定
        print(f"INFO: モデル '{MODEL_PATH}' を正常にロードしました。")

        # ★★デバッグのヒント★★: ロードされたモデルの最終出力層のサイズを確認
        # これがclass_labelsの数と一致することを確認してください。
        # 例: モデルがnn.Sequentialの最後の層がnn.Linearの場合
        if isinstance(model, nn.Module) and hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            if model.fc.out_features != 131:
                print(f"WARNING: ロードされたモデルの出力クラス数 ({model.fc.out_features}) が、期待されるクラス数 (131) と一致しません。")
        elif isinstance(model, nn.Module) and hasattr(model, 'features') and isinstance(model.features[-1], nn.Linear):
            if model.features[-1].out_features != 131:
                print(f"WARNING: ロードされたモデルの出力クラス数 ({model.features[-1].out_features}) が、期待されるクラス数 (131) と一致しません。")
        else:
            print("WARNING: モデルの出力層のクラス数を自動的に検証できませんでした。手動で確認してください。")

    else:
        print(f"WARNING: モデルファイル '{MODEL_PATH}' が見つかりません。ダミー応答モードで動作します。")
except Exception as e:
    model = None
    print(f"ERROR: モデル '{MODEL_PATH}' のロードに失敗しました: {str(e)}")
    print("INFO: ダミー応答モードで動作します。")

# 画像の前処理のための変換を定義
# ★★重要★★: これはモデルの訓練時に使用した変換と完全に一致させる必要があります。
# 訓練時に異なるリサイズや正規化を使用した場合、ここも修正してください。
transform = transforms.Compose([
    transforms.Resize((224, 224)), # モデルの入力サイズにリサイズ
    transforms.ToTensor(),         # PIL ImageをPyTorchテンソルに変換 (0-1に正規化される)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNetの統計情報で正規化
])


@app.get("/")
async def read_root():
    """ルートエンドポイント。FastAPIサーバーが動作していることを確認します。"""
    return {"message": "Hello from FastAPI!"}

@app.get("/plant_identify_test")
async def plant_identify_test():
    """APIテスト用のエンドポイント。"""
    return {"status": "success", "message": "API is working! Ready for grape identification."}

@app.post("/predict_grape/")
async def predict_grape(file: UploadFile = File(...)):
    """
    アップロードされた画像を処理し、ブドウの判定結果を返します。
    機械学習モデル 'grape_classifier.pt' を使用します。
    """
    # モデルがロードされていない場合はエラー応答を返す
    if model is None:
        return JSONResponse(
            status_code=503, # Service Unavailable
            content={
                "filename": file.filename,
                "identified_plant": "判定不可",
                "confidence": 0.0,
                "message": "モデルのロードに失敗したか、ファイルが見つかりません。システム管理者に連絡してください。"
            }
        )

    try:
        # ファイルの内容を非同期で読み込む
        contents = await file.read()
        # BytesIOを使ってPIL Imageとして開く
        # .convert("RGB") で確実にRGB形式に変換します (PNGのアルファチャンネルなどに対応)
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 画像を前処理
        input_tensor = transform(image)
        # モデルは通常バッチで入力されることを想定しているため、バッチ次元を追加 (例: [C, H, W] -> [1, C, H, W])
        input_batch = input_tensor.unsqueeze(0)

        # 推論を実行
        with torch.no_grad(): # 勾配計算を無効化 (推論時には不要でメモリを節約)
            output = model(input_batch)

        # モデルの出力を解釈
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 最も高い確率を持つクラスのインデックスを取得
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()

        # クラスラベルのマッピング (モデルの出力インデックスと一致させる必要があります)
        # ★★重要★★: このリストの順序は、モデルが訓練された際のクラスインデックスの順序と完全に一致する必要があります。
        # 提供された131クラスのリストを基に作成
        class_labels = {
            0: "Apple Braeburn",
            1: "Apple Crimson Snow",
            2: "Apple Golden 1",
            3: "Apple Golden 2",
            4: "Apple Golden 3",
            5: "Apple Granny Smith",
            6: "Apple Pink Lady",
            7: "Apple Red 1",
            8: "Apple Red 2",
            9: "Apple Red 3",
            10: "Apple Red Delicious",
            11: "Apple Red Yellow 1",
            12: "Apple Red Yellow 2",
            13: "Apricot",
            14: "Avocado",
            15: "Avocado ripe",
            16: "Banana",
            17: "Banana Lady Finger",
            18: "Banana Red",
            19: "Beetroot",
            20: "Blueberry",
            21: "Cactus fruit",
            22: "Cantaloupe 1",
            23: "Cantaloupe 2",
            24: "Carambula",
            25: "Cauliflower",
            26: "Cherry 1",
            27: "Cherry 2",
            28: "Cherry Rainier",
            29: "Cherry Wax Black",
            30: "Cherry Wax Red",
            31: "Cherry Wax Yellow",
            32: "Chestnut",
            33: "Clementine",
            34: "Cocos",
            35: "Corn",
            36: "Corn Husk",
            37: "Cucumber Ripe",
            38: "Cucumber Ripe 2",
            39: "Dates",
            40: "Eggplant",
            41: "Fig",
            42: "Ginger Root",
            43: "Granadilla",
            44: "Grape Blue",
            45: "Grape Pink",
            46: "Grape White",
            47: "Grape White 2",
            48: "Grape White 3",
            49: "Grape White 4",
            50: "Grapefruit Pink",
            51: "Grapefruit White",
            52: "Guava",
            53: "Hazelnut",
            54: "Huckleberry",
            55: "Kaki",
            56: "Kiwi",
            57: "Kohlrabi",
            58: "Kumquats",
            59: "Lemon",
            60: "Lemon Meyer",
            61: "Limes",
            62: "Lychee",
            63: "Mandarine",
            64: "Mango",
            65: "Mango Red",
            66: "Mangostan",
            67: "Maracuja",
            68: "Melon Piel de Sapo",
            69: "Mulberry",
            70: "Nectarine",
            71: "Nectarine Flat",
            72: "Nut Forest",
            73: "Nut Pecan",
            74: "Onion Red",
            75: "Onion Red Peeled",
            76: "Onion White",
            77: "Orange",
            78: "Papaya",
            79: "Passion Fruit",
            80: "Peach",
            81: "Peach 2",
            82: "Peach Flat",
            83: "Pear",
            84: "Pear 2",
            85: "Pear Abate",
            86: "Pear Forelle",
            87: "Pear Kaiser",
            88: "Pear Monster",
            89: "Pear Red",
            90: "Pear Stone",
            91: "Pear Williams",
            92: "Pepino",
            93: "Pepper Green",
            94: "Pepper Orange",
            95: "Pepper Red",
            96: "Pepper Yellow",
            97: "Physalis",
            98: "Physalis with Husk",
            99: "Pineapple",
            100: "Pineapple Mini",
            101: "Pitahaya Red",
            102: "Plum",
            103: "Plum 2",
            104: "Plum 3",
            105: "Pomegranate",
            106: "Pomelo Sweetie",
            107: "Potato Red",
            108: "Potato Red Washed",
            109: "Potato Sweet",
            110: "Potato White",
            111: "Quince",
            112: "Rambutan",
            113: "Raspberry",
            114: "Redcurrant",
            115: "Salak",
            116: "Strawberry",
            117: "Strawberry Wedge",
            118: "Tamarillo",
            119: "Tangelo",
            120: "Tomato 1",
            121: "Tomato 2",
            122: "Tomato 3",
            123: "Tomato 4",
            124: "Tomato Cherry Red",
            125: "Tomato Heart",
            126: "Tomato Maroon",
            127: "Tomato Yellow",
            128: "Tomato not Ripened",
            129: "Walnut",
            130: "Watermelon"
        }

        if predicted_class_idx in class_labels:
            identified_plant = class_labels[predicted_class_idx]
        else:
            identified_plant = "不明な植物 (インデックス範囲外)"

        return {
            "filename": file.filename,
            "identified_plant": identified_plant,
            "confidence": round(confidence, 4),
            "message": "機械学習モデルによる判定結果です。"
        }

    except Exception as e:
        print(f"ERROR: 画像処理またはモデル推論エラー: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"ファイル処理またはモデル推論エラー: {str(e)}"}
        )
