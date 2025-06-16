import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os

# --- Parametry ---
model_name = "bert-base-uncased"
model_weights_path = "model_save.bin"
MAX_LEN = 50
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class BERTLSTMClassifier(nn.Module):
    def __init__(self, bert_model, lstm_hidden_dim=128, lstm_layers=1, dropout_rate=0.3, bidirectional=True):
        super(BERTLSTMClassifier, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        fc_input_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = bert_output.last_hidden_state
        lstm_output, (hidden, _) = self.lstm(sequence_output)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        dropout_output = self.dropout(hidden)
        logits = self.fc(dropout_output)
        return logits

# --- Tokenizer and BERT ---
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
# --- Model ---
model = BERTLSTMClassifier(bert_model, lstm_hidden_dim=128, lstm_layers=2, bidirectional=True)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()

# --- Funkcja predykcji intensywności emocji ---
def predict_emotion_intensity(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        prob = torch.sigmoid(logits).item()
    # prawdopodobieństwo intensywności emocji (0-1)
    return prob

# --- Prosty loop czatu ---
def chat_loop():
    print("Chat z klasyfikacją intensywności emocji (wpisz 'quit' aby zakończyć):")
    while True:
        user_input = input("Ty: ")
        if user_input.lower() == 'quit':
            print("Koniec czatu.")
            break
        intensity = predict_emotion_intensity(user_input)
        print(f"Intensywność emocji (0-1): {intensity:.3f}")

if __name__ == "__main__":
    chat_loop()
