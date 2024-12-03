import numpy as np
import unittest
import torch
from data_dir.data_preparation import EmotionDataProcessor
from data_dir.dataset import EmotionDataset
from data_dir.model import EmotionTagger
from data_dir.training import ModelTrainer
from transformers import BertTokenizer
from torch.utils.data import DataLoader

class TestEmotionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Przygotowanie danych testowych i inicjalizacja obiektów
        emotion_map = {
            'neutral': 0, 'acceptance': 1, 'disgust': 2, 'surprise': 3,
            'joy': 4, 'sadness': 5, 'anger': 6, 'fear': 8,
            np.nan: None
        }
        cls.processor = EmotionDataProcessor(emotion_map=emotion_map, test_size=0.3, random_state=42)
        cls.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        cls.train_df, cls.val_df, cls.test_df, cls.mlb = cls.processor.process('D:/julixus/MEISD/meisd_project/MEISD/MEISD_text.csv')

        # Przygotowanie datasetów i dataloaderów
        cls.train_dataset = EmotionDataset(cls.train_df, cls.tokenizer, MAX_LEN=50)
        cls.val_dataset = EmotionDataset(cls.val_df, cls.tokenizer, MAX_LEN=50)
        cls.train_dataloader = DataLoader(cls.train_dataset, batch_size=8, shuffle=True)
        cls.val_dataloader = DataLoader(cls.val_dataset, batch_size=8)

        # Inicjalizacja modelu
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = EmotionTagger('bert-base-cased', num_classes=len(cls.mlb.classes_)).to(cls.device)
        cls.optimizer = torch.optim.AdamW(cls.model.parameters(), lr=2e-5)
        cls.loss_fn = torch.nn.BCELoss()
        cls.trainer = ModelTrainer(cls.model, cls.optimizer, cls.loss_fn, cls.device)

    def test_data_loading(self):
        """Test ładowania danych"""
        self.assertGreater(len(self.train_df), 0, "DataFrame trainingowy jest pusty")
        self.assertGreater(len(self.val_df), 0, "DataFrame walidacyjny jest pusty")
        self.assertGreater(len(self.test_df), 0, "DataFrame testowy jest pusty")

    def test_model_initialization(self):
        """Test poprawności inicjalizacji modelu"""
        self.assertIsInstance(self.model, EmotionTagger, "Model nie jest instancją klasy EmotionTagger")

    def test_training_step(self):
        """Test kroku treningowego"""
        train_accuracy, train_loss, train_f1 = self.trainer.train_epoch(self.train_dataloader)
        self.assertGreater(train_accuracy, 0.0, "Dokładność na treningu jest za niska")
        self.assertGreater(train_f1, 0.0, "F1-score na treningu jest za niski")

    def test_evaluation_step(self):
        """Test kroku ewaluacji"""
        val_accuracy, val_loss, val_f1 = self.trainer.eval_epoch(self.val_dataloader)
        self.assertGreater(val_accuracy, 0.0, "Dokładność na walidacji jest za niska")
        self.assertGreater(val_f1, 0.0, "F1-score na walidacji jest za niski")

    def test_model_forward_pass(self):
        """Test przepuszczenia danych przez model"""
        sample_input = self.train_df.iloc[0]['Utterances']
        encoded_input = self.tokenizer(sample_input, return_tensors='pt', padding=True, truncation=True, max_length=50).to(self.device)
        output = self.model(**encoded_input)
        self.assertEqual(output.shape[0], 1, "Output modelu ma nieprawidłowy kształt")
        self.assertEqual(output.shape[1], len(self.mlb.classes_), "Output modelu ma nieprawidłowy rozmiar")

if __name__ == '__main__':
    unittest.main()
