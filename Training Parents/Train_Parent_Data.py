import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class AnimalDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_excel(file_path)
        self.paths = self.data['Path'].values
        self.labels = self.data['Condition'].values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

def create_model_directory(input_excel_file, model_name, omit_batch_idx):
    base_dir = os.path.dirname(input_excel_file)
    model_dir = os.path.join(base_dir, 'models', model_name, f'batch_{omit_batch_idx+1}')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def initialize_model(model_name):
    if model_name == 'convnext':
        model = models.convnext_base(weights='DEFAULT')
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    elif model_name == 'efficientnet_v2':
        model = models.efficientnet_v2_s(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(weights='DEFAULT')
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif model_name == 'regnet':
        model = models.regnet_y_400mf(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, 2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, model_name, input_excel_file, omit_batch_idx, epochs=2000, patience=50):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    model_dir = create_model_directory(input_excel_file, model_name, omit_batch_idx)

    log_path = os.path.join(model_dir, f"{model_name}_training_log_batch_{omit_batch_idx+1}.txt")
    log_file = open(log_path, 'w')

    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
        precision_0 = report['0']['precision']
        recall_0 = report['0']['recall']
        f1_0 = report['0']['f1-score']
        precision_1 = report['1']['precision']
        recall_1 = report['1']['recall']
        f1_1 = report['1']['f1-score']

        epoch_time = time.time() - start_time
        log_print(f"Epoch [{epoch+1}/{epochs}]")
        log_print(f"Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
        log_print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
        log_print(f"Validation Precision (Class 0): {precision_0:.4f}, Recall (Class 0): {recall_0:.4f}, F1-Score (Class 0): {f1_0:.4f}")
        log_print(f"Validation Precision (Class 1): {precision_1:.4f}, Recall (Class 1): {recall_1:.4f}, F1-Score (Class 1): {f1_1:.4f}")
        log_print(f"Time for Epoch: {epoch_time:.2f} seconds")
        log_print('-' * 50)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}_best.pth"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log_print("Early stopping triggered!")
            break

    log_file.close()

    # Save Loss Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Curve (Batch {omit_batch_idx+1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, f"{model_name}_loss_plot_batch_{omit_batch_idx+1}.png"))
    plt.close()

    # Save Accuracy Plot
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy Curve (Batch {omit_batch_idx+1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, f"{model_name}_accuracy_plot_batch_{omit_batch_idx+1}.png"))
    plt.close()

    return train_losses, val_losses

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
    return cm, report

# Main loop
file_paths = [
    r'J:\paper2\Parents\Batch_1_YF12749_0_YF12811_1.xlsx',
    r'J:\paper2\Parents\Batch_2_YF12892_0_OF174_1.xlsx',
    r'J:\paper2\Parents\Batch_3_YF12825B_0_OF62_1.xlsx',
    r'J:\paper2\Parents\Batch_4_YF12876B_0_YF13921_1.xlsx',
    r'J:\paper2\Parents\Batch_5_YF12612_0_YF12746_1.xlsx',
    r'J:\paper2\Parents\Batch_6_YF12752_0_YF12750_1.xlsx'
]

models_to_train = ['convnext', 'efficientnet_v2', 'mobilenet_v3', 'regnet', 'vit']

for model_name in models_to_train:
    results = []
    combined_cm = np.zeros((2, 2))
    confusion_matrices = []

    for leave_out_idx in range(len(file_paths)):
        train_files = [file_paths[i] for i in range(len(file_paths)) if i != leave_out_idx]
        test_file = file_paths[leave_out_idx]

        train_dataset = torch.utils.data.ConcatDataset([AnimalDataset(f, transform) for f in train_files])
        test_dataset = AnimalDataset(test_file, transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = initialize_model(model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"Training {model_name} (leave out batch {leave_out_idx + 1})...")
        train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, model_name, test_file, leave_out_idx, epochs=2000, patience=50)

        cm, report = evaluate_model(model, test_loader)
        combined_cm += cm
        confusion_matrices.append((leave_out_idx + 1, cm))

        accuracy = report['accuracy']
        precision_0 = report['0']['precision']
        recall_0 = report['0']['recall']
        f1_0 = report['0']['f1-score']
        precision_1 = report['1']['precision']
        recall_1 = report['1']['recall']
        f1_1 = report['1']['f1-score']

        results.append([leave_out_idx + 1, accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1])

    output_file = os.path.join(os.path.dirname(file_paths[0]), f"{model_name}_evaluation.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        df_results = pd.DataFrame(results, columns=["Omitted", "Accuracy", "Prec_0", "Rec_0", "F1_0", "Prec_1", "Rec_1", "F1_1"])
        df_results.to_excel(writer, sheet_name="Metrics", index=False)

        for idx, (batch_num, cm) in enumerate(confusion_matrices):
            title = f"Confusion Matrix Batch {batch_num}"
            cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["True_0", "True_1"])
            cm_df.to_excel(writer, sheet_name=title, index=True)

    print(f"Results saved for {model_name} at {output_file}")
