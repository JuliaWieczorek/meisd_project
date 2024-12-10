from transformers import BertModel
import torch
import torch.nn as nn

class EmotionTagger(nn.Module):
    def __init__(self, model_name, num_classes):
        super(EmotionTagger, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.dropout(output.pooler_output)
        return torch.sigmoid(self.classifier(x))
