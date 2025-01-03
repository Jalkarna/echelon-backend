import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig, AdamW
from typing import List

class SignGrammarDataset(Dataset):
    """
    Custom dataset class to support sign-language grammar fine-tuning.
    The dataset should contain examples of English text (input) and 
    sign-language-formatted text (target).
    """
    def __init__(self, file_path: str, tokenizer: LlamaTokenizer, max_length: int = 128):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # Each item has an "input_text" and a "sign_text" 
                # or similar fields you define for the training process.
                src_text = item["input_text"]
                tgt_text = item["sign_text"]
                # We'll create a combined input for causal LM modeling
                combined = f"Input: {src_text}\nSign: {tgt_text}"
                self.examples.append(combined)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        # Tokenize
        tokenized = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "labels": input_ids,  # for causal LM, labels = input_ids
        }

def fine_tune(
    train_file: str, 
    model_save_dir: str = "saved_model", 
    epochs: int = 1, 
    batch_size: int = 2, 
    lr: float = 2e-5,
    max_length: int = 128
):

    os.makedirs(model_save_dir, exist_ok=True)

    # Attempt to download automatically if "llama-3.2-3b" not locally available
    model_name = "meta-llama/Llama-3.2-3B"
    # For a private/gated model, ensure you 'huggingface-cli login' or pass a token.
    token_str = "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"  # Insert your Hugging Face token
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=token_str)

    train_dataset = SignGrammarDataset(train_file, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If not found locally, this downloads from HF Hub (assuming permission / correct name)
    model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=token_str).to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Model saved to {model_save_dir}")

def load_sign_grammar_model(model_dir: str = "saved_model"):
    """
    Load the fine-tuned LLaMA model and tokenizer for sign-language grammar.
    """
    # If the user saved to model_dir, load from there. Otherwise, it tries to fetch from HF.
    # e.g. model_dir could be "llama-3.2-3b" or a local folder with the trained weights.
    model_dir_or_name = model_dir
    tokenizer = LlamaTokenizer.from_pretrained(model_dir_or_name)
    model = LlamaForCausalLM.from_pretrained(model_dir_or_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def convert_text_to_sign_grammar(input_text: str, model, tokenizer, max_length: int = 128) -> str:
    """
    Use the fine-tuned language model to convert English text to sign-grammar text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    prompt = f"Input: {input_text}\nSign:"
    tokenized = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            tokenized, 
            max_length=max_length,
            num_beams=5,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Remove the entire prompt portion if present
    if "Sign:" in generated_text:
        sign_text = generated_text.split("Sign:")[-1]
    else:
        sign_text = generated_text

    # Strip out leftover <pad> tokens or repeated [PAD]
    sign_text = sign_text.replace("<pad>", "").replace("[PAD]", "")
    # Also trim whitespace
    sign_text = sign_text.strip()

    return sign_text 