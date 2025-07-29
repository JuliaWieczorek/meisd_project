"""
Poprawiony BERT-LSTM Binary Classifier z regularizacją przeciw przetrenowaniu:
- Zwiększone regularyzacje (dropout, weight decay)
- Wczesne zatrzymywanie na podstawie validation loss
- Proper gradient clipping
- Zamrożenie większej części BERT na początku
- Kontrola learning rate
- POPRAWIONE: epochs handling, ESConv config, early stopping logic
"""

import os
import time
import json
import random
import sys
import datetime
from pathlib import Path
import csv


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

DEFAULTS = {
    "max_len": 32, #100,
    "batch_size": 4,  #16,  # Zmniejszono batch size
    "epochs_MEISD": 1, #15,
    "epochs_ESConv": 1, #3,
    "learning_rate_meisd": 5e-6,  # Zmniejszono learning rate
    "learning_rate_esconv": 2e-6,  # Jeszcze mniejszy dla fine-tuningu
    "dropout": 0.5,  # Zwiększono dropout
    "bert_dropout": 0.3,  # Dodano dropout dla BERT
    "bert_model": "bert-base-cased",
    "random_seed": 42,
    "loss_type": "focal",
    "focal_alpha": 2,  # Zmniejszono alfa
    "focal_gamma": 1.5,  # Zmniejszono gamma
    "lstm_hidden_dim": 64,  # Zmniejszono rozmiar LSTM
    "lstm_layers": 1,  # Zmniejszono liczbę warstw
    "scheduler_patience": 3,  # Zwiększono patience
    "scheduler_factor": 0.3,  # Więcej redukcji LR
    "scheduler_mode": "min",
    "early_stopping_patience": 3,  # Wczesne zatrzymywanie
    "early_stopping_patience_esconv": 1,  # DODANE: Osobna patience dla ESConv
    "weight_decay": 0.01,  # L2 regularization
    "gradient_clip": 1.0,  # Gradient clipping
    "freeze_bert_layers": 10,  # Zamróż więcej warstw BERT
    "log_dir": "logs",
    "output_dir": "./outputs"
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class BERTLSTMClassifier(nn.Module):
    def __init__(self, bert_model, lstm_hidden_dim=64, lstm_layers=1,
                 dropout_rate=0.5, bert_dropout=0.3, bidirectional=True):
        super(BERTLSTMClassifier, self).__init__()

        # Dodaj dropout do BERT
        self.bert = bert_model
        self.bert.config.hidden_dropout_prob = bert_dropout
        self.bert.config.attention_probs_dropout_prob = bert_dropout

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        # Dodaj więcej regularizacji
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)  # Drugi dropout layer

        fc_input_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim

        # Dodaj batch normalization
        self.batch_norm = nn.BatchNorm1d(fc_input_dim)

        # Zmniejsz rozmiar hidden layer
        self.fc1 = nn.Linear(fc_input_dim, fc_input_dim // 2)
        self.fc2 = nn.Linear(fc_input_dim // 2, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Inicjalizacja wag dla lepszej stabilności"""
        for module in [self.fc1, self.fc2]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = bert_output.last_hidden_state

        lstm_output, (hidden, _) = self.lstm(sequence_output)

        if self.lstm.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Zastosuj regularizację
        x = self.dropout1(hidden)
        x = self.batch_norm(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)

        return logits

def freeze_bert_layers(model, freeze_until_layer=10):
    """Zamróź więcej warstw BERT dla lepszej regularizacji"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if "bert.encoder.layer" in name:
            try:
                layer_num = int(name.split('.')[3])
            except:
                continue
            if layer_num < freeze_until_layer:
                param.requires_grad = False
                frozen_count += 1
        elif "bert.embeddings" in name:
            # Zamróź też embeddings
            param.requires_grad = False
            frozen_count += 1

    print(f"Frozen {frozen_count} BERT parameters (layers 0-{freeze_until_layer-1} + embeddings)")

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets.float())
        probas = torch.sigmoid(inputs)
        p_t = targets * probas + (1 - targets) * (1 - probas)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def load_data(file_path, dataset_type="ESConv"):
    """
    Wczytuje dane z różnych formatów i mapuje labele na binarne

    Args:
        file_path: ścieżka do pliku
        dataset_type: "ESConv" lub "MEISD" dla odpowiedniego mapowania
    """
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        conversation, labels = [], []
        for dialog in data:
            for turn in dialog.get("dialog", []):
                if turn.get("speaker") == "seeker":
                    text = turn.get("content", "").strip()
                    if 10 < len(text) < 200:
                        conversation.append(text)
                        # Dla JSON zakładamy że to ESConv bez konkretnych labelek
                        # W tym przypadku wszystkie będą 0 (low intensity)
                        # TODO: dopisac dopieranie odpowiednio labelek
                        # labels.append(0)
        df = pd.DataFrame({"conversation": conversation, "label": labels})
    else:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

    df = df[['conversation', 'label']].dropna()

    # Mapowanie na binarne labele
    unique_labels = df['label'].unique()
    is_already_binary = set(unique_labels).issubset({0, 1})

    if is_already_binary:
        print(f"Labels are already binary: {sorted(unique_labels)}")
        print(f"Current label distribution: {df['label'].value_counts().to_dict()}")
    else:
        print(f"Original labels: {sorted(unique_labels)}")

        if dataset_type == "ESConv":
            # ESConv (1-5) → Binary: 1-3 = 0 (low), 4-5 = 1 (high)
            df['label'] = df['label'].apply(lambda x: 0 if x <= 3 else 1)
            print(f"ESConv binary mapping applied: 1-3 → 0 (low), 4-5 → 1 (high)")
        elif dataset_type == "MEISD":
            # MEISD (1-3) → Binary: ≤1.5 = 0 (low), >1.5 = 1 (high)
            df['label'] = df['label'].apply(lambda x: 0 if x <= 1.5 else 1)
            print(f"MEISD binary mapping applied: ≤1.5 → 0 (low), >1.5 → 1 (high)")

        print(f"Final label distribution: {df['label'].value_counts().to_dict()}")

    return df
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, config, tag):
    if tag == "MEISD":
        epochs = config["epochs_MEISD"]
        early_stopping_patience = config["early_stopping_patience"]
    elif tag == "ESConv":
        epochs = config["epochs_ESConv"]
        early_stopping_patience = config["early_stopping_patience_esconv"]
    else:
        epochs = config.get("epochs", 10)  # fallback
        early_stopping_patience = config["early_stopping_patience"]

    print(f"Training {tag} for {epochs} epochs with early stopping patience {early_stopping_patience}")

    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    logs = []

    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).float().to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) >= 0.5).long()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

        log_line = f"[{tag}] Epoch {epoch+1}/{epochs}: TLoss={train_loss:.4f}, TAcc={train_acc:.4f}, TF1={train_f1:.4f} | VLoss={val_loss:.4f}, VAcc={val_acc:.4f}, VF1={val_f1:.4f}"
        print(log_line)
        logs.append(log_line)

        # Model saving and early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{config['output_dir']}/best_model_{tag}.pt")
            patience_counter = 0
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        scheduler.step(val_loss)

        # POPRAWIONE: Lepsze early stopping dla ESConv
        if tag == "ESConv":
            if val_acc >= 0.98:
                print(f"Stopping early on ESConv: val_acc >= 0.98 ({val_acc:.4f})")
                break
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping on ESConv at epoch {epoch+1} (patience reached)")
                break
        else:
            # Early stopping dla MEISD
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping on {tag} at epoch {epoch+1} (patience reached)")
                break

        # Dodatkowa informacja o learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if epoch == 0 or current_lr != optimizer.param_groups[0]['lr']:
            print(f"Current learning rate: {current_lr:.2e}")

    print(f"Training {tag} completed. Best validation loss: {best_val_loss:.4f}, Best F1: {best_f1:.4f}")
    return logs

def evaluate(model, loader, device, tag, config=None):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).float().to(device)
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    probs_array = np.array(all_probs).flatten()
    print(f"\nProbability distribution for {tag}:")
    print(f"Mean: {probs_array.mean():.4f}, Std: {probs_array.std():.4f}")
    print(f"Min: {probs_array.min():.4f}, Max: {probs_array.max():.4f}")

    report = classification_report(all_labels, all_preds, digits=4)
    with open(f"{tag}_report.txt", "w") as f:
        f.write(f"Probability Statistics:\n")
        f.write(f"Mean: {probs_array.mean():.4f}, Std: {probs_array.std():.4f}\n")
        f.write(f"Min: {probs_array.min():.4f}, Max: {probs_array.max():.4f}\n\n")
        f.write(report)

    print(report)
    # Pobierz teksty (musimy mieć je pod ręką!)
    if hasattr(loader.dataset, 'texts'):
        texts = loader.dataset.texts
    else:
        texts = [""] * len(all_labels)  # fallback

    # Zapisz predykcje i uruchom analizę
    df = save_predictions(texts, all_labels, all_preds, all_probs, tag, output_dir=config["output_dir"])
    analyze_predictions(df, tag, output_dir=config["output_dir"])

    return report

def save_predictions(texts, labels, preds, probs, tag, output_dir="outputs"):
    df = pd.DataFrame({
        "text": texts,
        "label": np.array(labels).astype(int).flatten(),
        "prediction": np.array(preds).astype(int).flatten(),
        "confidence": probs,
        "correct": [int(p == l) for p, l in zip(preds, labels)]
    })
    df.to_csv(f"{output_dir}/{tag}_predictions.csv", index=False)
    print(f"Saved predictions to {output_dir}/{tag}_predictions.csv")
    return df

def analyze_predictions(df, tag, output_dir="outputs"):
    cm = confusion_matrix(df["label"], df["prediction"])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {tag}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{tag}_confusion_matrix.png")
    plt.close()
    print(f"Saved confusion matrix to {output_dir}/{tag}_confusion_matrix.png")

    # Błędne przykłady
    fp = df[(df["label"] == 0) & (df["prediction"] == 1)].head(5)
    fn = df[(df["label"] == 1) & (df["prediction"] == 0)].head(5)
    error_examples = pd.concat([fp, fn])
    error_examples.to_csv(f"{output_dir}/{tag}_error_examples.csv", index=False)
    print(f"Saved 5 FP and FN examples to {output_dir}/{tag}_error_examples.csv")


def run_pipeline(meisd_path, esconv_path, config):
    set_seed(config["random_seed"])
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loss function
    if config["loss_type"] == "focal":
        loss_fn = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
        print(f"Using Focal Loss (alpha={config['focal_alpha']}, gamma={config['focal_gamma']})")
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        print("Using BCE Loss")

    # Model initialization
    bert_model = BertModel.from_pretrained(config["bert_model"], output_hidden_states=True)
    model = BERTLSTMClassifier(
        bert_model=bert_model,
        lstm_hidden_dim=config["lstm_hidden_dim"],
        lstm_layers=config["lstm_layers"],
        dropout_rate=config["dropout"],
        bert_dropout=config["bert_dropout"]
    ).to(device)

    # Freeze BERT layers
    freeze_bert_layers(model, freeze_until_layer=config["freeze_bert_layers"])

    def prepare_loaders(df, tag=""):
        # Stratified split dla lepszego podziału klas
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                df['conversation'], df['label'],
                test_size=0.2,
                random_state=config["random_seed"],
                stratify=df['label'] if len(df['label'].unique()) > 1 else None
            )
        except ValueError:
            # Jeśli stratify nie działa, użyj zwykłego split
            X_train, X_val, y_train, y_val = train_test_split(
                df['conversation'], df['label'],
                test_size=0.2,
                random_state=config["random_seed"]
            )

        print(f"{tag} - Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"{tag} - Train labels distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"{tag} - Val labels distribution: {pd.Series(y_val).value_counts().to_dict()}")

        train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer, config["max_len"])
        val_dataset = TextDataset(X_val.tolist(), y_val.tolist(), tokenizer, config["max_len"])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
        return train_loader, val_loader

    # Phase 1: Train on MEISD
    print("\n" + "="*50)
    print("Phase 1: Training on MEISD")
    print("="*50)
    meisd_df = load_data(meisd_path, dataset_type="MEISD")
    print(f"MEISD dataset size: {len(meisd_df)}")
    print(f"MEISD label distribution: {meisd_df['label'].value_counts().to_dict()}")

    train_loader, val_loader = prepare_loaders(meisd_df, "MEISD")
    optimizer = torch.optim.AdamW(  # AdamW zamiast Adam dla lepszej regularizacji
        model.parameters(),
        lr=config["learning_rate_meisd"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler_mode"],
        patience=config["scheduler_patience"],
        factor=config["scheduler_factor"]
    )

    logs = train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, device, config, tag="MEISD")

    # Load best model and evaluate
    print("\n" + "-"*30)
    print("Evaluating best MEISD model:")
    print("-"*30)
    model.load_state_dict(torch.load(f"{config['output_dir']}/best_model_MEISD.pt"))
    evaluate(model, val_loader, device, tag="MEISD", config=config)

    # Phase 2: Fine-tune on ESConv
    print("\n" + "="*50)
    print("Phase 2: Fine-tuning on ESConv")
    print("="*50)
    esconv_df = load_data(esconv_path, dataset_type="ESConv")
    print(f"ESConv dataset size: {len(esconv_df)}")
    print(f"ESConv label distribution: {esconv_df['label'].value_counts().to_dict()}")

    # Unfreeze some BERT layers for fine-tuning
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if "bert.encoder.layer" in name:
            try:
                layer_num = int(name.split('.')[3])
                if layer_num >= 8:  # Unfreeze last 4 layers
                    param.requires_grad = True
                    unfrozen_count += 1
            except:
                continue
    print(f"Unfrozen {unfrozen_count} BERT parameters for fine-tuning (layers 8-11)")

    train_loader, val_loader = prepare_loaders(esconv_df, "ESConv")

    # POPRAWIONE: Nowy optimizer z mniejszym learning rate dla fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate_esconv"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler_mode"],
        patience=config["scheduler_patience"],
        factor=config["scheduler_factor"],
    )

    # POPRAWIONE: Przekazanie konfiguracji z odpowiednimi parametrami dla ESConv
    logs += train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, device, config, tag="ESConv")

    # Load best model and evaluate
    print("\n" + "-"*30)
    print("Evaluating best ESConv model:")
    print("-"*30)
    model.load_state_dict(torch.load(f"{config['output_dir']}/best_model_ESConv.pt"))
    evaluate(model, val_loader, device, tag="ESConv", config=config)

    # Save logs
    log_file = f"{config['output_dir']}/training_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Training Log\n")
        f.write("="*50 + "\n")
        f.write(f"Configuration used:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\n".join(logs))

    print(f"\nTraining logs saved to: {log_file}")
    print("Training completed successfully!")

def run_esconv_finetune_only(esconv_path, meisd_model_path, config):
    """
    Uruchom tylko fine-tuning na ESConv używając wcześniej wytrenowanego modelu MEISD

    Args:
        esconv_path: ścieżka do poprawnego datasetu ESConv
        meisd_model_path: ścieżka do zapisanego modelu MEISD (best_model_MEISD.pt)
        config: konfiguracja (użyj DEFAULTS)
    """
    set_seed(config["random_seed"])
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loss function
    if config["loss_type"] == "focal":
        loss_fn = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
        print(f"Using Focal Loss (alpha={config['focal_alpha']}, gamma={config['focal_gamma']})")
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        print("Using BCE Loss")

    # Model initialization
    bert_model = BertModel.from_pretrained(config["bert_model"], output_hidden_states=True)
    model = BERTLSTMClassifier(
        bert_model=bert_model,
        lstm_hidden_dim=config["lstm_hidden_dim"],
        lstm_layers=config["lstm_layers"],
        dropout_rate=config["dropout"],
        bert_dropout=config["bert_dropout"]
    ).to(device)

    # Wczytaj wytrenowany model MEISD
    print(f"Loading pre-trained MEISD model from: {meisd_model_path}")
    if not os.path.exists(meisd_model_path):
        raise FileNotFoundError(f"MEISD model not found at: {meisd_model_path}")

    model.load_state_dict(torch.load(meisd_model_path, map_location=device))
    print("MEISD model loaded successfully!")

    # Unfreeze some BERT layers for fine-tuning
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if "bert.encoder.layer" in name:
            try:
                layer_num = int(name.split('.')[3])
                if layer_num >= 8:  # Unfreeze last 4 layers
                    param.requires_grad = True
                    unfrozen_count += 1
            except:
                continue
    print(f"Unfrozen {unfrozen_count} BERT parameters for fine-tuning (layers 8-11)")

    def prepare_loaders(df, tag=""):
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                df['conversation'], df['label'],
                test_size=0.2,
                random_state=config["random_seed"],
                stratify=df['label'] if len(df['label'].unique()) > 1 else None
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                df['conversation'], df['label'],
                test_size=0.2,
                random_state=config["random_seed"]
            )

        print(f"{tag} - Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"{tag} - Train labels distribution: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"{tag} - Val labels distribution: {pd.Series(y_val).value_counts().to_dict()}")

        train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), tokenizer, config["max_len"])
        val_dataset = TextDataset(X_val.tolist(), y_val.tolist(), tokenizer, config["max_len"])
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
        return train_loader, val_loader

    # Fine-tune na ESConv
    print("\n" + "="*50)
    print("Fine-tuning on ESConv (using pre-trained MEISD model)")
    print("="*50)
    esconv_df = load_data(esconv_path, dataset_type="ESConv")
    print(f"ESConv dataset size: {len(esconv_df)}")
    print(f"ESConv label distribution: {esconv_df['label'].value_counts().to_dict()}")

    # Sprawdź czy mamy różne labelki
    unique_labels = esconv_df['label'].unique()
    if len(unique_labels) <= 1:
        print(f"WARNING: Dataset has only {len(unique_labels)} unique label(s): {unique_labels}")
        print("This might not be suitable for binary classification training!")
        return

    train_loader, val_loader = prepare_loaders(esconv_df, "ESConv")

    # Optimizer z mniejszym learning rate dla fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate_esconv"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler_mode"],
        patience=config["scheduler_patience"],
        factor=config["scheduler_factor"],
    )

    # Użyj istniejącej funkcji train_model z parametrem tag="ESConv"
    logs = train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, device, config, tag="ESConv")

    # Load best model and evaluate
    print("\n" + "-"*30)
    print("Evaluating best ESConv fine-tuned model:")
    print("-"*30)
    model.load_state_dict(torch.load(f"{config['output_dir']}/best_model_ESConv.pt"))
    metrics = evaluate(model, val_loader, device, tag="ESConv_finetune", config=config)

    append_to_summary(
        tag="ESConv_finetune_only",
        dataset_name="ESConv",
        metrics=metrics,
        model_path=f"{config['output_dir']}/best_model_ESConv.pt",
        output_dir=config["output_dir"]
    )

    # Save logs
    log_file = f"{config['output_dir']}/esconv_finetune_only_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("ESConv Fine-tuning Only Log\n")
        f.write("="*50 + "\n")
        f.write(f"Configuration used:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nPre-trained model loaded from: {meisd_model_path}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\n".join(logs))

    print(f"\nFine-tuning logs saved to: {log_file}")
    print("ESConv fine-tuning completed successfully!")


def append_to_summary(tag, dataset_name, metrics, model_path, output_dir):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_dir = Path(output_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = summary_dir / "summary.csv"
    summary_json = summary_dir / "summary.json"
    summary_xlsx = summary_dir / "summary.xlsx"

    # Prepare one row of results
    record = {
        "tag": tag,
        "dataset": dataset_name,
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "timestamp": timestamp,
        "model_path": str(model_path),
    }

    # Append to CSV
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    # Append to JSON
    data = []
    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as jf:
            try:
                data = json.load(jf)
            except json.JSONDecodeError:
                data = []

    data.append(record)
    with open(summary_json, "w", encoding="utf-8") as jf:
        json.dump(data, jf, indent=4)

    # Write Excel (overwrite each time from JSON)
    df = pd.DataFrame(data)
    df.to_excel(summary_xlsx, index=False)


def run_full_experiment(config, variants):
    meisd_dir = "C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation"
    esconv_path = "C:/Users/juwieczo/DataspellProjects/meisd_project/data/esconv_both_parts.csv"

    for variant_name, meisd_filename in variants.items():
        print(f"\n\n{'='*80}")
        print(f"▶▶▶ Processing variant: {variant_name}")
        print(f"{'='*80}")

        meisd_path = os.path.join(meisd_dir, meisd_filename)

        # Skopiuj config, żeby nie nadpisywać
        local_config = config.copy()

        # Ustaw odpowiednie ścieżki wyjściowe
        local_config["output_dir"] = f"./outputs/{variant_name}"
        os.makedirs(local_config["output_dir"], exist_ok=True)

        # Uruchom pipeline dla tej wersji augmentacji
        run_pipeline(meisd_path, esconv_path, local_config)

        # Przenieś modele do unikalnych nazw
        for dataset in ["MEISD", "ESConv"]:
            src = f"{local_config['output_dir']}/best_model_{dataset}.pt"
            dst = f"{local_config['output_dir']}/best_model_{variant_name}_{dataset}.pt"
            if os.path.exists(src):
                os.rename(src, dst)
                print(f"Renamed model: {src} → {dst}")

if __name__ == "__main__":
    config = DEFAULTS.copy()

    print("Running full experiment across all augmentation variants...")
    variants = {
        #"LLM":        "esconv_enhanced_llm_augmentation_70percent_balanced.xlsx",
        "Mixed":      "esconv_enhanced_mixed_augmentation_70percent_balanced.xlsx",
        "NLP":        "esconv_enhanced_nlp_augmentation_70percent_balanced.xlsx",
        "NLP-LLM":    "esconv_enhanced_llm_nlp_augmentation_70percent_balanced.xlsx",
        "Classical":  "esconv_enhanced_classical_augmentation_70percent_balanced.xlsx"
    }

    try:
        run_full_experiment(config, variants)
    except Exception as e:
        print(f"\n⚠️ Error during full experiment: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n✅ All training completed successfully!")
    current_hour = datetime.datetime.now().hour

    # Warunek: między północą a 6:59 rano
    if 0 <= current_hour < 7:
        print("Jest między 00:00 a 07:00 – komputer zostanie wyłączony za 1 minutę...")
        time.sleep(60)
        os.system("shutdown /s /t 1")
    else:
        print("🕒 Koniec analizy.")
