from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ===== Google Sheets Setup =====
SERVICE_ACCOUNT_FILE = 'credentials.json'  # File kredensial dari Google Cloud
SPREADSHEET_ID = '1GM9mlRGoUTNu0APh_J7v-tXZgUozNpInFOKiijA8B98'  # ID dari link spreadsheet kamu
RANGE_NAME = 'Sheet1!A1'  # Atur sheet dan range awal (bisa disesuaikan)
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

def append_to_sheet(user_id, rekomendasi, scores):
    service = build('sheets', 'v4', credentials=credentials)
    values = [[str(user_id)] + [str(r) for r in rekomendasi] + [str(s) for s in scores]]
    body = {'values': values}
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()

# ===== Model Kustom =====
class MatrixFactorization(tf.keras.Model):
    def __init__(self, n_users, n_places, embedding_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.n_users = n_users
        self.n_places = n_places
        self.embedding_dim = embedding_dim

        self.user_embedding = tf.keras.layers.Embedding(n_users, embedding_dim)
        self.place_embedding = tf.keras.layers.Embedding(n_places, embedding_dim)

    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:, 0])
        place_vec = self.place_embedding(inputs[:, 1])
        return tf.reduce_sum(user_vec * place_vec, axis=1)

    def get_config(self):
        return {
            "n_users": self.n_users,
            "n_places": self.n_places,
            "embedding_dim": self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ===== Load Model =====
model = load_model("model_rekomendasi.keras", custom_objects={"MatrixFactorization": MatrixFactorization})

# ===== Flask App =====
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or "user_id" not in data or "candidates" not in data:
        return jsonify({"error": "Harap kirim 'user_id' dan 'candidates'"}), 400

    user_id = data["user_id"]
    candidates = data["candidates"]

    input_data = np.array([[user_id, place_id] for place_id in candidates])
    predictions = model.predict(input_data, verbose=0)

    top_indices = predictions.argsort()[-3:][::-1]
    rekomendasi = [candidates[i] for i in top_indices]
    scores = [float(predictions[i]) for i in top_indices]

    # Simpan ke Google Sheets
    append_to_sheet(user_id, rekomendasi, scores)

    return jsonify({
        "user_id": user_id,
        "rekomendasi_tempat_id": rekomendasi,
        "scores": scores
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "API Rekomendasi Tempat sudah berjalan.",
        "usage": {
            "POST /predict": {
                "body": {
                    "user_id": "int",
                    "candidates": "[list of int]"
                }
            }
        }
    })

# Tidak perlu app.run() agar Hugging Face Spaces bisa menjalankan otomatis
