import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import numpy as np
from datetime import datetime
import logging
from typing import Tuple, Optional, List
from torch.cuda.amp import autocast, GradScaler

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DATA_DIR = '/Users/nicholastanner/Documents/garbage-training-dataset'
    TEST_DATA_DIR = '/Users/nicholastanner/Documents/garbage-testing-dataset'
    CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 0.01
    NUM_CLASSES = len(CLASSES)
    NUM_WORKERS = 2
    PATIENCE = 15
    SAVE_PATH = os.path.join(BASE_DIR, 'trash_classifier.pth')
    USE_MIXED_PRECISION = True

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomAffine(
        degrees=30,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2)
    ),
    transforms.RandomPerspective(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TrashDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None, class_mapping: Optional[dict] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                self.class_names.append(item)

        self.class_names.sort()

        if class_mapping:
            self.class_to_idx = class_mapping
        else:
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in image_files:
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[index]
        label = self.labels[index]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label

class TrashClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(TrashClassifier, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        for layer in [self.model.layer4]:
            for param in layer.parameters():
                param.requires_grad = True

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def prepare_data(config: Config) -> Tuple[Dataset, Dataset]:
    class_mapping = {class_name: idx for idx, class_name in enumerate(config.CLASSES)}

    train_dataset = TrashDataset(
        data_dir=config.TRAIN_DATA_DIR,
        transform=train_transform,
        class_mapping=class_mapping
    )

    test_dataset = TrashDataset(
        data_dir=config.TEST_DATA_DIR,
        transform=test_transform,
        class_mapping=class_mapping
    )

    return train_dataset, test_dataset

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[optim.lr_scheduler._LRScheduler], device: torch.device, epoch: int, scaler: Optional[GradScaler] = None) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{Config.NUM_EPOCHS}')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        if np.random.random() > 0.5:
            mixed_x, target_a, target_b, lam = mixup_data(images, labels)

            if scaler is not None:
                with autocast():
                    outputs = model(mixed_x)
                    loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
            else:
                outputs = model(mixed_x)
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        else:
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    return running_loss / len(train_loader), 100. * correct / total

@torch.no_grad()
def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return test_loss / len(test_loader), 100. * correct / total

def main(config: Config) -> nn.Module:
    logging.info(f"Starting training at {datetime.now()}")

    torch.set_float32_matmul_precision('high')

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device for accelerated training on M4 Pro")
    else:
        device = torch.device("cpu")
        logging.info("MPS not available, using CPU")

    train_dataset, test_dataset = prepare_data(config)
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    logging.info(f"Training classes: {train_dataset.class_names}")
    logging.info(f"Testing classes: {test_dataset.class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )

    model = TrashClassifier(config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    params = [
        {'params': model.model.fc.parameters(), 'lr': config.LEARNING_RATE},
        {'params': model.model.layer4.parameters(), 'lr': config.LEARNING_RATE * 0.1}
    ]

    optimizer = optim.AdamW(
        params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.LEARNING_RATE, config.LEARNING_RATE * 0.1],
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    scaler = GradScaler() if config.USE_MIXED_PRECISION else None

    logging.info("Starting training...")
    best_acc = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler
        )
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        logging.info(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}:')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            best_model_state = model.state_dict()
            logging.info(f"New best accuracy: {best_acc:.2f}% - Model saved")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logging.info(f"Early stopping triggered after {config.PATIENCE} epochs without improvement")
                break

        logging.info('-' * 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logging.info(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")
    return model

def save_final_model(model: nn.Module, config: Config) -> None:
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': config.NUM_CLASSES,
        'class_names': config.CLASSES,
    }, config.SAVE_PATH)
    logging.info(f"\nFinal model saved to {config.SAVE_PATH}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = Config()
    model = main(config)
    save_final_model(model, config)
