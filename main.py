from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"

print("🔄 Loading RuBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = ["negative", "neutral", "positive"]

def analyze_sentiment(text: str):
    """
    Возвращает:
      label: str — метка тональности
      prob: float — уверенность модели
    """
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = model(**encoded)

    probs = torch.softmax(output.logits, dim=1)[0]
    label_id = torch.argmax(probs).item()
    return LABELS[label_id], float(probs[label_id])

from pydantic import BaseModel
from typing import List

class TextsRequest(BaseModel):
    texts: List[str]

class SentimentItem(BaseModel):
    text: str
    sentiment: str
    confidence: float

class SentimentResponse(BaseModel):
    results: List[SentimentItem]

from fastapi import FastAPI

app = FastAPI(
    title="Russian Sentiment API (RuBERT)",
    description="Тональность текстов на русском через RuBERT",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "message": "RuBERT sentiment service is running"}

@app.post("/sentiment", response_model=SentimentResponse)
def sentiment_endpoint(req: TextsRequest):
    results = []
    for text in req.texts:
        label, conf = analyze_sentiment(text)
        results.append(SentimentItem(
            text=text,
            sentiment=label,
            confidence=conf
        ))
    return SentimentResponse(results=results)
