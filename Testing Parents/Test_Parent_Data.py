import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === Dataset ===
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

# === Model Initializer ===
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
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)

# === Evaluation ===
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
    return cm, report

# === Configuration ===
model_names = ['convnext', 'efficientnet_v2', 'mobilenet_v3', 'regnet', 'vit']
num_batches = 9
model_base_path = r'J:\paper1\models'
test_base_path = r'J:\F2-Sows\Testing'
output_path = os.path.join(model_base_path, "evaluation_results.xlsx")

# === Run Evaluation ===
results = []
confusion_matrices = []

for model_name in model_names:
    for batch_idx in range(1, num_batches + 1):
        model_path = os.path.join(model_base_path, model_name, f'batch_{batch_idx}', f'{model_name}_best.pth')
        test_file = os.path.join(test_base_path, f'Batch_{batch_idx}.xlsx')

        if not os.path.exists(model_path) or not os.path.exists(test_file):
            print(f"[SKIP] Missing: {model_path} or {test_file}")
            continue

        print(f"[EVALUATE] {model_name} on Batch {batch_idx}")
        model = initialize_model(model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))

        test_dataset = AnimalDataset(test_file, transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        cm, report = evaluate_model(model, test_loader)
        cm_title = f"{model_name}_B{batch_idx}_CM"
        confusion_matrices.append((cm_title, cm))

        results.append([
            model_name, f"Batch_{batch_idx}", report['accuracy'],
            report['0']['precision'], report['0']['recall'], report['0']['f1-score'],
            report['1']['precision'], report['1']['recall'], report['1']['f1-score']
        ])

# === Save to Excel ===
with pd.ExcelWriter(output_path) as writer:
    # Default metrics sheet (model-wise)
    df = pd.DataFrame(results, columns=[
        "Model", "Batch", "Accuracy",
        "Prec_0", "Rec_0", "F1_0", "Prec_1", "Rec_1", "F1_1"
    ])
    df.to_excel(writer, sheet_name="Metrics", index=False)

    # Reorganized metrics sheet (batch-wise)
    df_sorted = df.copy()
    df_sorted['Batch_Num'] = df_sorted['Batch'].str.extract(r'Batch_(\d+)').astype(int)
    df_sorted = df_sorted.sort_values(by=['Batch_Num', 'Model'])
    df_sorted.drop(columns=['Batch_Num'], inplace=True)
    df_sorted.to_excel(writer, sheet_name="Metrics_By_Batch", index=False)

    # Save confusion matrices
    for name, cm in confusion_matrices:
        cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["True_0", "True_1"])
        cm_df.to_excel(writer, sheet_name=name)

print(f"[DONE] Results saved to: {output_path}")
