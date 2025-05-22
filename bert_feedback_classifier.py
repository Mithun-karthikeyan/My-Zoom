import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_df = pd.read_csv("train - train.csv")
eval_df = pd.read_csv("evaluation - evaluation.csv")

# Clean text
def clean_text(text):
    if pd.isnull(text): return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

for col in ["text", "reason"]:
    train_df[col] = train_df[col].apply(clean_text)
    eval_df[col] = eval_df[col].apply(clean_text)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Dataset class
class FeedbackDataset(Dataset):
    def __init__(self, texts, reasons, labels):
        self.encodings = tokenizer(texts, reasons, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = FeedbackDataset(train_df["text"].tolist(), train_df["reason"].tolist(), train_df["label"].tolist())
eval_dataset = FeedbackDataset(eval_df["text"].tolist(), eval_df["reason"].tolist(), eval_df["label"].tolist())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop (just 3 quick epochs)
for epoch in range(3):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Metrics
acc = accuracy_score(true_labels, preds)
prec, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="binary")

print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save model
model.save_pretrained("./feedback_model")
tokenizer.save_pretrained("./feedback_model")
