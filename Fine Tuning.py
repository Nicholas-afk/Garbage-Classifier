import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAIN_DIR = os.path.join(BASE_DIR, 'More training', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'More training', 'test')

os.makedirs(MODELS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

ORIGINAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trash_classifier_finetuned_20250405_003634_best.pth')
FINE_TUNED_MODEL_PATH = os.path.join(MODELS_DIR, f'trash_classifier_finetuned_{timestamp}.pth')

ORIGINAL_CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'random']

CLASS_MAPPING = {
    'hazardous': 'battery',
    'biological': 'biological',
    'cardboard': 'cardboard',
    'metal': 'metal',
    'plastic': 'plastic',
    'glass': 'glass',
    'recyclable': 'recyclable'
}

CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'random', 'recyclable']

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def print_status(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

class TrashDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        print_status(f"Initializing dataset from {data_dir}...")
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        available_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

        print_status(f"Found folders in dataset: {', '.join(available_folders)}")

        for folder in available_folders:
            if folder in CLASS_MAPPING:
                class_name = CLASS_MAPPING[folder]
                folder_path = os.path.join(data_dir, folder)

                class_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                print_status(f"Found {len(class_images)} images for '{folder}' (mapped to '{class_name}')")

                for img_name in class_images:
                    self.images.append(os.path.join(folder_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])
            else:
                print_status(f"Warning: Folder '{folder}' not in class mapping, skipping")

        print_status(f"Total dataset size: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print_status(f"Error loading {image_path}: {str(e)}")
            return torch.zeros((3, 224, 224)), self.labels[idx]

class TrashClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(TrashClassifier, self).__init__()
        print_status("Initializing model...")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def save_model(model, optimizer, epoch, accuracy, is_best=False):
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'num_classes': len(CLASSES),
        'class_names': CLASSES,
        'fine_tuned_by': "Nicholas-afk"
    }

    torch.save(save_dict, FINE_TUNED_MODEL_PATH)
    print_status(f"Saved fine-tuned model to: {FINE_TUNED_MODEL_PATH}")

    if is_best:
        best_path = FINE_TUNED_MODEL_PATH.replace('.pth', '_best.pth')
        torch.save(save_dict, best_path)
        print_status(f"Saved best model to: {best_path}")

def train_model():
    try:
        print_status("Setting up data transforms...")
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
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = TrashDataset(TRAIN_DIR, transform=train_transform)
        val_dataset = TrashDataset(TEST_DIR, transform=val_transform)

        print_status(f"Training set size: {len(train_dataset)}")
        print_status(f"Validation set size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = TrashClassifier(num_classes=len(CLASSES)).to(DEVICE)

        original_checkpoint = torch.load(ORIGINAL_MODEL_PATH, map_location=DEVICE)

        model_dict = model.state_dict()
        orig_dict = {k: v for k, v in original_checkpoint['model_state_dict'].items() if k in model_dict and 'model.fc.4' not in k}
        model_dict.update(orig_dict)
        model.load_state_dict(model_dict, strict=False)

        print_status("Successfully loaded original model weights (except final classification layer)")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1, verbose=True)

        best_accuracy = 0.0
        patience = 6
        patience_counter = 0

        print_status("Starting training...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{running_loss / (pbar.n + 1):.3f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_accuracy = 100. * val_correct / val_total
            print_status(f'\nEpoch {epoch + 1} Results:')
            print_status(f'Training Loss: {running_loss / len(train_loader):.3f}')
            print_status(f'Training Accuracy: {100. * correct / total:.2f}%')
            print_status(f'Validation Loss: {val_loss / len(val_loader):.3f}')
            print_status(f'Validation Accuracy: {val_accuracy:.2f}%')

            scheduler.step(val_accuracy)

            is_best = val_accuracy > best_accuracy
            if is_best:
                best_accuracy = val_accuracy
                patience_counter = 0
                save_model(model, optimizer, epoch, val_accuracy, is_best=True)
                print_status(f'New best model saved! Accuracy: {val_accuracy:.2f}%')
            else:
                patience_counter += 1
                save_model(model, optimizer, epoch, val_accuracy, is_best=False)

            if patience_counter >= patience:
                print_status(f"Early stopping triggered after {patience} epochs without improvement")
                break

        print_status(f"Training completed. Best accuracy: {best_accuracy:.2f}%")

    except Exception as e:
        print_status(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    print_status("Starting fine-tuning process...")
    print_status(f"Original model: {ORIGINAL_MODEL_PATH}")
    print_status(f"Fine-tuned model will be saved as: {FINE_TUNED_MODEL_PATH}")
    print_status(f"Using device: {DEVICE}")
    print_status(f"Using training dataset from: {TRAIN_DIR}")
    print_status(f"Using validation dataset from: {TEST_DIR}")
    print_status(f"New class mapping: {CLASS_MAPPING}")
    print_status(f"Total classes: {len(CLASSES)} - {', '.join(CLASSES)}")

    try:
        train_model()
    except Exception as e:
        print_status(f"Fine-tuning failed: {str(e)}")
    else:
        print_status("Fine-tuning completed successfully")
        print_status(f"Final model saved to: {FINE_TUNED_MODEL_PATH}")
