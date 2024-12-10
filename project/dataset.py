from torch.utils.data import Dataset
import torch


class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        utterance = data_row['Utterances']
        labels = data_row.iloc[1:].values.astype(float)  # Zakładamy, że kolumny etykiet są od drugiej kolumny

        encoding = self.tokenizer(
            utterance,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float),
        }
