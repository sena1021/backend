from fastapi import FastAPI, File, UploadFile
from PIL import Image # 画像処理のライブラリは残しておくと良い
import io

app = FastAPI()

# --- PyTorchモデルのロード部分をコメントアウトまたは削除 ---
# モデルはコメントアウトされたままで、ダミー応答を返す設定を維持します。
# model = None
# try:
#     model = torch.load("models/plant_classifier.pt")
#     model.eval()
#     print("INFO: PyTorchモデルが正常にロードされました。")
# except Exception as e:
#     print(f"ERROR: 機械学習モデルのロードに失敗しました: {e}")
#     model = None
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# --- ここまでコメントアウト ---


@app.get("/")
async def read_root():
    """ルートエンドポイント。FastAPIサーバーが動作していることを確認します。"""
    return {"message": "Hello from FastAPI!"}

@app.get("/plant_identify_test")
async def plant_identify_test():
    """APIテスト用のエンドポイント。"""
    return {"status": "success", "message": "API is working! Ready for grape identification."} # メッセージをブドウ用に更新

# --- /predict_grape/ エンドポイントをブドウ判定用のダミー応答に修正 ---
@app.post("/predict_grape/") # エンドポイント名をpredict_grapeに変更しました
async def predict_grape(file: UploadFile = File(...)): # 関数名もpredict_grapeに変更しました
    """
    アップロードされた画像を処理し、ブドウのダミー判定結果を返します。
    現在の実装では機械学習モデルは使用せず、ダミー応答を返します。
    """
    try:
        # 受け取ったファイル名を確認します
        filename = file.filename
        
        # 実際にはここで機械学習モデルを使った判定処理が入りますが、
        # 今回はダミー応答を返します。
        
        return {
            "filename": filename,
            "identified_plant": "ダミー判定: ブドウ", # ダミーの植物名をブドウに変更
            "confidence": 0.90, # ダミーの信頼度を任意で変更
            "message": "モデルが未実装のため、ダミー結果を返します。これはブドウの判定システム用です。" # メッセージを更新
        }

    except Exception as e:
        # ファイル処理中にエラーが発生した場合のハンドリング
        print(f"ERROR: ファイル処理エラー: {str(e)}") # サーバーログにエラーを出力
        return {"error": f"ファイル処理エラー: {str(e)}"}