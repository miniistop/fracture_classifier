import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import time
from pathlib import Path
import copy

class FractureDataModule:
    def __init__(self, data_path: str, batch_size: int = 128, num_workers: int = 4, seed: int = 42):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.class_names = {
            "0": "Normal",
            "1": "Fracture"
        }
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def setup(self):
        # Load dataset
        dataset = datasets.ImageFolder(self.data_path, transform=self.transform)
        
        # Split dataset
        train_idx, temp_idx = train_test_split(
            list(range(len(dataset))),
            test_size=0.2,
            random_state=self.seed
        )
        
        val_idx, test_idx = train_test_split(
            list(range(len(temp_idx))),
            test_size=0.5,
            random_state=self.seed
        )
        
        # Create subsets
        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, [temp_idx[i] for i in val_idx])
        self.test_dataset = Subset(dataset, [temp_idx[i] for i in test_idx])
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

class FractureClassifier:
    def __init__(self, model_name: str = 'efficientnet-b4', lr: float = 0.05):
        self.model = EfficientNet.from_pretrained(model_name, num_classes=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lr_lambda=lambda epoch: 0.98739
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def fit(self, data_module, num_epochs: int = 300):
        best_acc = 0.0
        best_model = None
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(data_module.train_dataloader())
            val_loss, val_acc = self.validate(data_module.val_dataloader())
            self.scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(self.model.state_dict())
                
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
        
        # Load best model
        self.model.load_state_dict(best_model)
        return history
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history['train_acc'], label='Train')
    ax1.plot(history['val_acc'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history['train_loss'], label='Train')
    ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Usage example:
if __name__ == "__main__":
    # Initialize data module
    data_module = FractureDataModule(data_path='Data/Train')
    data_module.setup()
    
    # Initialize model
    model = FractureClassifier()
    
    # Train model
    history = model.fit(data_module)
    
    # Plot results
    plot_training_history(history)
    
    # Save model
    model.save_model('fracture_model.pt')