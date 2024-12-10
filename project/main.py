import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_preparation import EmotionDataProcessor
from dataset import EmotionDataset
from model import EmotionTagger
from training import ModelTrainer
from tqdm import tqdm
import time


# Hyperparameters
MODEL_NAME = 'bert-base-cased'
BATCH_SIZE = 16
MAX_LEN = 50
EPOCHS = 25
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.001

# Data preparation
emotion_map = {
    'neutral': 0, 'acceptance': 1, 'disgust': 2, 'surprise': 3,
    'joy': 4, 'sadness': 5, 'anger': 6, 'fear': 8,
    np.nan: None
}
print(f'''Hyperparameters:
        MODEL_NAME: {MODEL_NAME}, 
        BATCH_SIZE: {BATCH_SIZE}, 
        MAX_LEN F1: {MAX_LEN},
        EPOCHS: {EPOCHS},
        LEARNING_RATE: {LEARNING_RATE},
        WEIGHT_DECAY: {WEIGHT_DECAY}''')



processor = EmotionDataProcessor(emotion_map=emotion_map, test_size=0.3, random_state=42)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_df, val_df, test_df, mlb = processor.process('D:/julixus/data/meisd_project/data/MEISD_text.csv')
train_dataset = EmotionDataset(train_df, tokenizer, MAX_LEN)
val_dataset = EmotionDataset(val_df, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionTagger(MODEL_NAME, num_classes=len(mlb.classes_)).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.BCELoss()

trainer = ModelTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    # Train
    train_accuracy, train_loss, train_f1 = trainer.train_epoch(train_dataloader)
    print(f'Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
    # Validate
    val_accuracy, val_loss, val_f1 = trainer.eval_epoch(val_dataloader)
    print(f'Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation F1: {val_f1:.4f}')


#%%
