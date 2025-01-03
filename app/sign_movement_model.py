import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from typing import List

class MovementDataset(Dataset):
    """
    Takes sign_text and frames_json. We'll treat this as a sequence generation problem.
    Example item:
      sign_text: "WHAT UP?"
      frames_json: "[...]"
    We feed a prompt: "Sign: WHAT UP?\nFrames:" => want model to generate => "[ ... ]"
    """
 
    def __init__(self, file_path: str, tokenizer: GPT2Tokenizer, max_length: int = 256):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                prompt = f"Sign: {item['sign_text']}\nFrames: {item['frames_json']}"
                self.examples.append(prompt)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)     # shape: (max_length)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }

def fine_tune_movement(
    train_file: str,
    model_save_dir: str = "saved_movement_model",
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_length: int = 256
):
    """
    Fine-tune GPT2 to produce frames JSON from sign_text.
    Expects data/training_data_movements.json or similar.
    """
    os.makedirs(model_save_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Add a [PAD] token so we can do padding
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    train_dataset = MovementDataset(train_file, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[MovementModel] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"[MovementModel] Saved to {model_save_dir}")

def load_movement_model(model_dir: str = "saved_movement_model"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    # If not set, define pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def convert_sign_text_to_frames(sign_text: str, model, tokenizer, max_length: int = 256) -> str:
    """
    Use the fine-tuned movement model to generate frames JSON from sign_text.
    Model sees prompt:  "Sign: xxxxxx\nFrames:" => outputs  "[{...}]" 
    We parse or directly return that frames JSON string.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    prompt = f"Sign: {sign_text}\nFrames:"
    encoded = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            encoded,
            max_length=max_length,
            num_beams=5,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # The text might contain the entire prompt "Sign: xxxxx\nFrames:" plus JSON
    # Example: "Sign: HELLO YOU?\nFrames: [ { ... } ]"
    # Let's isolate the bracketed frames with a simple approach:
    if "Frames:" in generated_text:
        frames_part = generated_text.split("Frames:")[-1].strip()
    else:
        frames_part = generated_text

    return frames_part 