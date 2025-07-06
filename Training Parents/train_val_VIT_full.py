import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import ViTForImageClassification, ViTFeatureExtractor


# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT models typically use 224x224 images
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load training data from the first Excel file
train_file_path = r'F:\videos\train_set.xlsx'
df_train = pd.read_excel(train_file_path)

# Load validation data from the second Excel file
val_file_path = r'F:\videos\test_set.xlsx'
df_val = pd.read_excel(val_file_path)

# Split data paths and labels for training and validation datasets
train_image_paths = df_train['Path'].tolist()
train_labels = df_train['Condition'].tolist()

val_image_paths = df_val['Path'].tolist()
val_labels = df_val['Condition'].tolist()

# Create datasets
train_dataset = CustomDataset(train_image_paths, train_labels, transform=train_transforms)
val_dataset = CustomDataset(val_image_paths, val_labels, transform=train_transforms)

# Print number of images in training and validation sets
print(f"Number of images in training set: {len(train_dataset)}")
print(f"Number of images in validation set: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the ViT model with pre-trained weights (without the classifier)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Replace the classifier layer to match the number of labels (2 in this case)
model.config.num_labels = 2
model.classifier = nn.Linear(model.config.hidden_size, model.config.num_labels)
model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
model.classifier.bias.data.zero_()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a folder for saving models
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_folder = os.path.join(os.path.dirname(train_file_path), 'save_models', f'VIT(train_combined)_full_dataset_{timestamp}')
os.makedirs(save_folder, exist_ok=True)

best_val_accuracy = 0.0
best_model_path = None
patience = 500  # Early stopping patience
early_stopping_counter = 0

num_epochs = 5000

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train_preds = 0
    total_train_preds = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total_train_preds += labels.size(0)
        correct_train_preds += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_train_preds / total_train_preds
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

    # Evaluate the model
    model.eval()
    val_running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = correct_preds / total_preds

    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

    # Save the model if it is the best so far
    if val_accuracy > best_val_accuracy:
        # Delete the previous best model and its associated text file if they exist
        if best_model_path is not None:
            os.remove(best_model_path)
            metrics_file_path = best_model_path.replace('.pth', '.txt')
            if os.path.exists(metrics_file_path):
                os.remove(metrics_file_path)

        best_val_accuracy = val_accuracy
        best_model_path = os.path.join(save_folder, f'best_model_epoch_{epoch + 1}_acc_{val_accuracy:.4f}.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model to {best_model_path}')

        # Save metrics for the new best model
        metrics_file_path = best_model_path.replace('.pth', '.txt')
        with open(metrics_file_path, 'w') as f:
            f.write(f"Epoch: {epoch + 1}\n")
            f.write(f"Training Loss: {epoch_loss:.4f}\n")
            f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
            f.write(f"Validation Loss: {val_loss:.4f}\n")
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")
        
        # Reset early stopping counter
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Early stopping
    if early_stopping_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# Save the model from the last epoch if early stopping didn't trigger
if early_stopping_counter < patience:
    last_epoch_model_path = os.path.join(save_folder, f'final_model_epoch_{num_epochs}_acc_{val_accuracy:.4f}.pth')
else:
    last_epoch_model_path = os.path.join(save_folder, f'final_model_epoch_{epoch + 1}_acc_{val_accuracy:.4f}.pth')
    
torch.save(model.state_dict(), last_epoch_model_path)
print(f'Saved model from last epoch to {last_epoch_model_path}')
    
# Save metrics for the final model
final_metrics_file_path = last_epoch_model_path.replace('.pth', '.txt')
with open(final_metrics_file_path, 'w') as f:
    f.write(f"Final Model\n")
    f.write(f"Epoch: {epoch + 1}\n")
    f.write(f"Training Loss: {epoch_loss:.4f}\n")
    f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")

print("Training on full dataset complete.\n")
