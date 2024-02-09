import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

train_df = pd.read_csv("train.csv")

train_df["text"] = (
    train_df["title"] + " " + train_df["abstract"] + " " + train_df["body"]
)
train_df["text"] = train_df["text"].fillna("")
print(train_df.count())

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def parallel_encode_data(tokenizer, texts, max_length=512, chunk_size=1000):
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    input_ids = []
    attention_masks = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(encode_data, tokenizer, chunk, max_length)
            for chunk in chunks
        ]

        for future in as_completed(futures):
            ids, masks = future.result()
            input_ids.append(ids)
            attention_masks.append(masks)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def encode_data(tokenizer, texts, max_length=512):
    encoded_batch = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return encoded_batch["input_ids"], encoded_batch["attention_mask"]


train_inputs, train_masks = parallel_encode_data(tokenizer, train_df["text"].values)

train_labels = torch.tensor(train_df["subgroup"].factorize()[0])
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(train_df["subgroup"].unique())
).to("cuda")
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
model.train()
for step, batch in enumerate(train_dataloader):
    batch = tuple(t.to("cuda") for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    outputs = model(
        b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels
    )
    loss = outputs[0]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

test_df = pd.read_csv("test.csv")
test_df["text"] = test_df["title"] + " " + test_df["abstract"] + " " + test_df["body"]
test_df["text"] = test_df["text"].fillna("")
test_inputs, test_masks = parallel_encode_data(tokenizer, test_df["text"].values)

test_data = TensorDataset(test_inputs, test_masks)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

model.eval()
predictions = []
with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        b_input_ids, b_input_mask = batch

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]

        _, predicted_labels = torch.max(logits, dim=1)
        predictions.extend(predicted_labels.tolist())

predicted_subgroups = [
    list(train_df["subgroup"].factorize()[1])[i] for i in predictions
]
output_df = pd.DataFrame(predicted_subgroups, columns=["subgroup"])
output_df.to_csv("output.csv", index=False)
