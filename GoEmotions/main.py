import torch
import uvicorn
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from torchtext.data import get_tokenizer


class CheckEmotionText(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, output_dim=28):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.3)
        self.lin = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.lin(hidden)


NAMES = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
         'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
         'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
         'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
         'pride', 'realization', 'relief', 'remorse', 'sadness',
         'surprise', 'neutral']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = torch.load("vocab_GoEmotions.pth", map_location=device, weights_only=False)
model = CheckEmotionText(len(vocab)).to(device)
model.load_state_dict(torch.load("model_GoEmotions.pth", map_location=device))
model.eval()

app = FastAPI()
tokenizer = get_tokenizer("basic_english")


class TextIn(BaseModel):
    text: str


def preprocess(text: str):
    tokens = tokenizer(text)
    ids = [vocab[token] for token in tokens]
    return torch.tensor([ids], dtype=torch.int64, device=device)


@app.post("/predict")
def predict(item: TextIn, threshold: float = 0.5):
    x = preprocess(item.text)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0)

    results = {
        NAMES[i]: round(probs[i].item(), 4)
        for i in range(len(NAMES))
        if probs[i].item() > threshold
    }

    # Если ни одна эмоция не превысила порог — берём топ-1
    if not results:
        top_idx = probs.argmax().item()
        results = {NAMES[top_idx]: round(probs[top_idx].item(), 4)}

    return {"emotions": results}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
