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

def get_conv_output_size(model, input_size):
    with torch.no_grad():
        x = torch.zeros(1, 3, input_size, input_size)
        x = model.conv1(x)
        x = model.relu1(x)
        x = model.dropout1(x)
        x = model.conv2(x)
        x = model.relu2(x)
        x = model.maxpool1(x)
        x = model.conv3(x)
        x = model.relu3(x)
        x = model.dropout2(x)
        x = model.conv4(x)
        x = model.relu4(x)
        x = model.maxpool2(x)
        x = model.conv5(x)
        x = model.relu5(x)
        x = model.dropout3(x)
        x = model.conv6(x)
        x = model.relu6(x)
        x = model.maxpool3(x)
        x = model.flatten(x)
        return x.shape[1]

# Define the CNN model
class Net(nn.Module):
    def __init__(self, num_classes, conv_output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout4 = nn.Dropout(0.2)

        # Update fc1 input size based on calculated size
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.relu7 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.relu8 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dropout4(x)
        x = self.fc1(x)
        x = self.relu7(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.relu8(x)
        x = self.dropout6(x)
        x = self.fc3(x)
        return x

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your data from the Excel file
input_file_path = 'E:/FarmCare/pre_images.xlsx'
df_pre = pd.read_excel(input_file_path)

# Split data paths and labels
image_paths = df_pre['Path'].tolist()
labels = df_pre['Condition'].tolist()

# Define batch paths and labels
batch_paths = {
    'batch_1': [path for path in image_paths if 'B1_SOWS' in path or 'B2_SOWS' in path],
    'batch_2': [path for path in image_paths if 'B3_SOW' in path or 'B4_SOWS' in path],
    'batch_3': [path for path in image_paths if 'B5_SOWS' in path or 'B7_SOWS' in path],
    'batch_4': [path for path in image_paths if 'B6_SOWS' in path],
    'batch_5': [path for path in image_paths if 'B8_SOW' in path or 'B9_SOWS' in path]
}

batch_labels = {name: [labels[image_paths.index(path)] for path in paths] for name, paths in batch_paths.items()}

# Training parameters
num_epochs = 500
patience = 100  # Early stopping patience
early_stopping_counter = 0

# Create a folder for saving models
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_path = os.path.dirname(input_file_path)
save_base_folder = os.path.join(base_path, 'save_models')

for leave_out_batch in batch_paths.keys():
    print(f"Training with {leave_out_batch} as the validation set")

    # Prepare training and validation datasets
    train_paths = []
    train_labels = []
    for batch_name, paths in batch_paths.items():
        if (batch_name != leave_out_batch):
            train_paths.extend(paths)
            train_labels.extend(batch_labels[batch_name])
    
    val_paths = batch_paths[leave_out_batch]
    val_labels = batch_labels[leave_out_batch]
    
    train_dataset = CustomDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    # Print number of frames in training and validation sets
    print(f"Number of frames in training set: {len(train_dataset)}")
    print(f"Number of images in validation set: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create a model with a dummy conv_output_size
    dummy_model = Net(num_classes=2, conv_output_size=1)  # dummy size
    conv_output_size = get_conv_output_size(dummy_model, input_size=128)

    # Initialize the actual model with the computed conv_output_size
    model = Net(num_classes=2, conv_output_size=conv_output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create a folder for saving models for this specific leave-out batch
    save_folder = os.path.join(save_base_folder, f'CNN_{leave_out_batch}_{timestamp}')
    os.makedirs(save_folder, exist_ok=True)

    best_val_accuracy = 0.0
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train_preds = 0
        total_train_preds = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
                outputs = model(images)
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
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            # Delete the previous best model and its text file if they exist
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f'Deleted previous best model: {best_model_path}')
                
                previous_metrics_file_path = best_model_path.replace('.pth', '.txt')
                if os.path.exists(previous_metrics_file_path):
                    os.remove(previous_metrics_file_path)
                    print(f'Deleted previous best model metrics file: {previous_metrics_file_path}')
                
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
        torch.save(model.state_dict(), last_epoch_model_path)
        print(f'Saved model from last epoch to {last_epoch_model_path}')
        
        # Save metrics for the final model
        final_metrics_file_path = last_epoch_model_path.replace('.pth', '.txt')
        with open(final_metrics_file_path, 'w') as f:
            f.write(f"Final Model\n")
            f.write(f"Epoch: {num_epochs}\n")
            f.write(f"Training Loss: {epoch_loss:.4f}\n")
            f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
            f.write(f"Validation Loss: {val_loss:.4f}\n")
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")

    print(f"Training for leave-out batch {leave_out_batch} complete.\n")
