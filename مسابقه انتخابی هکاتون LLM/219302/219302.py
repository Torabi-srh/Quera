from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np

class TimeDirectionDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len):
        self.dataset = pd.read_csv(filepath)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        text = row["paragraph 1"] + " [SEP] " + row["paragraph 2"]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        label = 1 if row["label"] == "correct" else 0
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train(model, train_loader, val_loader, device, epochs=3):
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        total_train_loss = 0
        for batch in train_loader:
            b_input_ids = batch["input_ids"].to(device)
            b_attention_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)
            model.zero_grad()
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_attention_mask,
                labels=b_labels,
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch + 1}, Loss: {total_train_loss / len(train_loader)}")

        eval_accuracy = 0
        nb_eval_steps = 0
        model.eval()
        for batch in val_loader:
            b_input_ids = batch["input_ids"].to(device)
            b_attention_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)
            with torch.no_grad():
                outputs = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_attention_mask
                )

            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            tmp_eval_accuracy = accuracy_score(label_ids, np.argmax(logits, axis=1))
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps}")


def predict_and_save_output(model, test_loader, device, output_file="output.csv"):
    model.eval()
    predictions = []

    for batch in test_loader:
        b_input_ids = batch["input_ids"].to(device)
        b_attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_attention_mask
            )

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        predictions.extend(preds)

    predictions_labels = ["correct" if pred == 1 else "reverse" for pred in predictions]

    pd.DataFrame(predictions_labels, columns=["prediction"]).to_csv(
        output_file, index=False
    )


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = TimeDirectionDataset("train.csv", tokenizer, max_len=512)
val_dataset = TimeDirectionDataset("val.csv", tokenizer, max_len=512)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train(model, train_loader, val_loader, device)
