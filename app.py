from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Инициализация FastAPI
app = FastAPI()

# Инициализация модели и токенизатора
MODEL_PATH = "improved_trained_model.pth"
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Классы меток
LABELS = ['норма', 'лёгкое', 'серьёзное']

# Модель данных для API
class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(input: TextInput):
    try:
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
        return {"text": input.text, "predicted_label": LABELS[predicted_label]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def root():
    return {"message": "API работает!"}