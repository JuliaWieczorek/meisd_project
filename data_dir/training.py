import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        """
        Initializes the ModelTrainer.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer to use for training.
            loss_fn (torch.nn.Module): Loss function for training.
            device (torch.device): Device (CPU or GPU) where the model and data will be located.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, dataloader):
        """
        Trains the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): The DataLoader that provides the training data.

        Returns:
            float: Accuracy of the model on the current epoch.
            float: Average loss for the current epoch.
            float: F1 score of the model on the current epoch.
        """
        self.model.train()
        total_loss, correct, num_samples = 0, 0, 0
        all_preds, all_labels = [], []

        # Use tqdm for progress bar
        for batch in tqdm(dataloader, desc="Training", leave=False):
            input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).all(dim=1).sum().item()
            num_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

        accuracy = correct / num_samples
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return accuracy, avg_loss, f1

    def eval_epoch(self, dataloader):
        """
        Evaluates the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): The DataLoader that provides the validation/test data.

        Returns:
            float: Accuracy of the model on the current epoch.
            float: Average loss for the current epoch.
            float: F1 score of the model on the current epoch.
        """
        self.model.eval()
        total_loss, correct, num_samples = 0, 0, 0
        all_preds, all_labels = [], []

        # Use tqdm for progress bar
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.loss_fn(outputs, labels)

                preds = (outputs > 0.5).float()
                correct += (preds == labels).all(dim=1).sum().item()
                num_samples += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()

        accuracy = correct / num_samples
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return accuracy, avg_loss, f1
