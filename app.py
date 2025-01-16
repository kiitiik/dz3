from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import logging
import requests

from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
MODEL_PATH = "improved_trained_model.pth"
MODEL_URL = "https://example.com/improved_trained_model.pth"  # Замените на вашу ссылку
LABELS = ['норма', 'лёгкое', 'серьёзное']

# Инициализация FastAPI
app = FastAPI()

# Добавление CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Можно ограничить список доменов
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для модели и токенизатора
model = None
tokenizer = None

# Модель данных для API
class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_resources():
    global model, tokenizer

    # Проверка и загрузка модели
    if not os.path.exists(MODEL_PATH):
        logger.info("Модель не найдена локально. Загрузка...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        logger.info("Модель успешно загружена.")

    try:
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Модель успешно загружена в память.")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise RuntimeError("Не удалось загрузить модель.")

    # Загрузка токенизатора
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    logger.info("Токенизатор успешно загружен.")

@app.get("/")
def root():
    return {"message": "API работает!"}

@app.post("/predict/")
def predict_sentiment(input: TextInput):
    try:
        logger.info(f"Получен текст для анализа: {input.text}")

        # Токенизация текста
        encoded = tokenizer.encode_plus(
            input.text,
            add_special_tokens=True,
            max_length=50,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        # Прогноз
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

        # Возврат результата
        result = {"text": input.text, "predicted_label": LABELS[predicted_label]}
        logger.info(f"Результат предсказания: {result}")
        return result
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса.")


