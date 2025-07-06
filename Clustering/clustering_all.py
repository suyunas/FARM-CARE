# this code:
#features extracted from 
# emoti model, resnet-18, vit

#Dimensionality reduction:
# pca, tsne, umap

#Clustering:
# kmeans, agglomerative, spectral, gmm

import os
import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import re
import matplotlib.lines as mlines
import mplcursors
from umap import UMAP
import itertools
from torchvision import models

# emoti model
class Net(nn.Module):
    def __init__(self, num_classes):
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
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)
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

# Define CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['Path']
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            return self.__getitem__((idx + 1) % len(self.data))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.data.iloc[idx]['Condition'], dtype=torch.long if hasattr(self, 'num_classes') and self.num_classes > 1 else torch.float32)


        if 'sow' in img_path:
            animal_id = re.search(r'YF\d+[A-Z]?|OF\d+[A-Z]?', img_path).group(0) if re.search(r'YF\d+[A-Z]?|OF\d+[A-Z]?', img_path) else 'Unknown'
        else:
            animal_id = re.search(r'RF\d+', img_path).group(0) if re.search(r'RF\d+', img_path) else 'Unknown'

        #animal_id = re.search(r'RF\d+', img_path).group(0) if re.search(r'RF\d+', img_path) else 'Unknown'
        image_name = os.path.basename(img_path)
        return image, label, animal_id, image_name

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
def load_data(excel_file, transform):
    data = pd.read_excel(excel_file)
    #filtered_data = data[data['Condition'] == data['prediction']]

    exclude_rf= [        
        # b1
        #'RF28', 'RF33', 'RF36', 'RF50', 'RF52', 'RF59', 'RF34', 'RF40', 'RF53', 'RF56', 'RF60', 'RF61'
        
        #b2
        #'RF72' , 'RF73' , 'RF74', 'RF75' , 'RF78' , 'RF79', 'RF83', 'RF85' , 'RF86' , 'RF88' , 'RF91' , 'RF92',
        
        #b3
        #'RF96',  'RF103', 'RF104', 'RF105', 'RF112' , 'RF93' ,  'RF98' , 'RF109' ,  'RF114' , 'RF118', 'RF119' , 'RF120' ,

        #b4
        #'RF123', 'RF124', 'RF125', 'RF126', 'RF127', 'RF128', 'RF129', 'RF131', 'RF132',
        
        #b5
        #'RF138', 'RF139', 'RF140', 'RF133', 'RF135', 'RF136',
        #b6
        #'RF147', 'RF154', 'RF160', 'RF167', 'RF169', 'RF173', 'RF142', 'RF143', 'RF151', 'RF152', 'RF155', 'RF163', 'RF158', 'RF165', 'RF170', 'RF174', 'RF175', 'RF177',
        #b7
        #'RF185','RF186', 'RF187', 'RF189', 'RF198', 'RF202', 'RF188', 'RF191', 'RF196', 'RF197', 'RF203', 'RF205',
        #b8
        #'RF218', 'RF219','RF220', 'RF221', 'RF222', 'RF224', 'RF227', 'RF228', 'RF229', 'RF233', 'RF234', 'RF235'

         #'RF111'
        'D0',
        'D1',
        'D2',
        'D3',
        #'D70', 
        'D90'
        ]

    #filtered_data = filtered_data[~filtered_data['Path'].str.contains('|'.join(exclude_rf))]
    filtered_data = data[~data['Path'].str.contains('|'.join(exclude_rf))]

    dataset = CustomDataset(filtered_data, transform=transform)
    if len(set(filtered_data['Condition'])) > 1:
        dataset.num_classes = len(set(filtered_data['Condition']))
    return dataset

# Load dataset and DataLoader

#val_excel_file = 'data_gilt_with_predictions_gilt_all.xlsx'
#val_excel_file = 'n_data_gilt_sow_all.xlsx'
#val_excel_file = 'n_data_piglet_sow_all.xlsx'
#val_excel_file = 'n2_data_f3piglet_sow_all.xlsx'
val_excel_file = r'J:\F2-Daughters\B1\image_condition.xlsx'
val_dataset = load_data(val_excel_file, transform)
val_data_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load models
vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
#vit_model.load_state_dict(torch.load('./farm_care_model/syed_model.pth', map_location=torch.device('cpu')),  strict=False)
vit_model.load_state_dict(torch.load(r'C:\Users\syedu\Desktop\Coding\acode\models\best_model_epoch_13_acc_0.6759.pth', map_location=torch.device('cpu')),  strict=False)
vit_model.eval()

resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.load_state_dict(torch.load(r'C:\Users\syedu\Desktop\Coding\acode\models\fine_tuned_model_resnet.pth', map_location=torch.device('cpu')))
resnet_model.eval()

emoti_model = Net(num_classes=2)
emoti_model.load_state_dict(torch.load(r'C:\Users\syedu\Desktop\Coding\acode\models\fine_tuned_emoti_model.pth', map_location=torch.device('cpu')))
emoti_model.eval()

# Define the output directory
output_dir = r'J:\F2-Daughters\B1\clustering_results'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist


def extract_features(data_loader, model, feature_extraction_layer=False):
    features = []
    labels = []
    animal_ids = []
    image_names = []

    # Determine model type
    is_resnet = isinstance(model, models.ResNet)
    
    for data, target, animal_id, image_name in data_loader:
        # For custom `Net` model
        if feature_extraction_layer and isinstance(model, Net):
            x = model.conv1(data)
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
            outputs = x  # Extract features after flatten layer

        # For Vision Transformer (ViT) model
        elif isinstance(model, timm.models.vision_transformer.VisionTransformer):
            outputs = model.forward_features(data)  # Extract features from ViT

        # For ResNet-18 model
        elif is_resnet:
            # Extract features from ResNet-18 before final classification layer
            with torch.no_grad():
                x = model.conv1(data)           # Initial convolution
                x = model.bn1(x)               # Batch norm
                x = model.relu(x)              # ReLU
                x = model.maxpool(x)           # Max pooling

                x = model.layer1(x)            # Residual blocks
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)

                x = model.avgpool(x)           # Average pooling
                x = torch.flatten(x, 1)        # Flatten the tensor
                outputs = x

        else:
            raise ValueError("Model type not supported")

        features.append(outputs.detach().cpu().numpy())
        labels.append(target.detach().cpu().numpy())
        animal_ids.extend(animal_id)
        image_names.extend(image_name)
    
    return np.concatenate(features), np.concatenate(labels), animal_ids, image_names



# Determine suitable number of PCA components
def get_pca_components(features):
    n_samples, n_features = features.shape
    return min(50, n_samples, n_features)  # Adjust based on the smallest dimension

# Extract features for all models
vit_features, vit_labels, vit_animal_ids, vit_image_names = extract_features(val_data_loader, vit_model)
vit_feat = vit_features.reshape(vit_features.shape[0], -1)
#print(f"vit_feat shape: {vit_feat.shape}")
pca = PCA(n_components= 50, random_state=42)
vit_pca_features = pca.fit_transform(vit_feat)

resnet_features, resnet_labels, resnet_animal_ids, resnet_image_names = extract_features(val_data_loader, resnet_model)
resnet_feat = resnet_features.reshape(resnet_features.shape[0], -1)
#print(f"resnet_feat shape: {resnet_feat.shape}")
pca = PCA(n_components=50, random_state=42)
resnet_pca_features = pca.fit_transform(resnet_feat)

emoti_features, emoti_labels, emoti_animal_ids, emoti_image_names = extract_features(val_data_loader, emoti_model, feature_extraction_layer=True)
emoti_feat = emoti_features.reshape(emoti_features.shape[0], -1)
#print(f"emoti_feat shape: {emoti_feat.shape}")
pca = PCA(n_components=50, random_state=42)
emoti_pca_features = pca.fit_transform(emoti_feat)

# Combine features for analysis
#combined_features = np.concatenate((vit_features, resnet_features, emoti_features), axis=1)
#combined_labels = vit_labels  # Assuming the labels are the same across all models
#combined_animal_ids = vit_animal_ids
#combined_image_names = vit_image_names

# Define clustering methods
clustering_methods = {
    'KMeans': KMeans(n_clusters=2),
    'Agglomerative': AgglomerativeClustering(n_clusters=2),
    'Spectral': SpectralClustering(n_clusters=2, affinity='nearest_neighbors'),
    'GMM': GaussianMixture(n_components=2)
    
}

def plot_clustering(features, labels, method_name, animal_ids, image_names, title):
    sns.set(style="white", rc={"figure.figsize": (12, 10)})

    # Define color palette for stress (1) and non-stress (0)
    # Add a default color for outliers or unknown labels
    color_palette = {0: 'green', 1: 'red', -1: 'gray'}  # Gray for outliers
    colors = [color_palette.get(label, 'gray') for label in labels]

    # t-SNE Clustering
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(features)

    fig_tsne, ax_tsne = plt.subplots(figsize=(12, 10))
    scatter_tsne = ax_tsne.scatter(tsne_features[:, 0], tsne_features[:, 1], c=colors, s=50, alpha=0.7)
    ax_tsne.set_title(f'{title}', fontsize=20, weight='bold')

    # Add legend for t-SNE outside the plot
    handles_tsne = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_palette[i], markersize=10) for i in [0, 1]]
    labels_legend_tsne = ['LS', 'SS']
    ax_tsne.legend(handles_tsne, labels_legend_tsne, loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=20)

    # Make x and y axis values bold
    ax_tsne.tick_params(axis='both', which='major', labelsize=14, width=2)
    ax_tsne.xaxis.label.set_weight('bold')
    ax_tsne.yaxis.label.set_weight('bold')
    for label in ax_tsne.get_xticklabels() + ax_tsne.get_yticklabels():
        label.set_fontsize(14)
        label.set_weight('bold')

    # Annotate hover for t-SNE
    annot_tsne = ax_tsne.annotate("", xy=(0,0), xytext=(20,20),
                                  textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                                  arrowprops=dict(arrowstyle="->"))
    annot_tsne.set_visible(False)

    def update_annot_tsne(ind):
        pos = scatter_tsne.get_offsets()[ind["ind"][0]]
        annot_tsne.xy = pos
        text = "{}".format(" ".join([image_names[n] for n in ind["ind"]]))
        annot_tsne.set_text(text)
        annot_tsne.get_bbox_patch().set_facecolor(color_palette.get(labels[ind["ind"][0]], 'gray'))
        annot_tsne.get_bbox_patch().set_alpha(0.4)

    def hover_tsne(event):
        vis = annot_tsne.get_visible()
        if event.inaxes == ax_tsne:
            cont, ind = scatter_tsne.contains(event)
            if cont:
                update_annot_tsne(ind)
                annot_tsne.set_visible(True)
                fig_tsne.canvas.draw_idle()
            else:
                if vis:
                    annot_tsne.set_visible(False)
                    fig_tsne.canvas.draw_idle()

    fig_tsne.canvas.mpl_connect("motion_notify_event", hover_tsne)
    fig_path = os.path.join(output_dir, f'{title}.png')
    plt.tight_layout()
    plt.savefig(fig_path, format='png', bbox_inches='tight')
    #plt.show()

    # UMAP Clustering
    umap = UMAP(n_components=2, random_state=42)
    umap_features = umap.fit_transform(features)

    fig_umap, ax_umap = plt.subplots(figsize=(20, 10))
    scatter_umap = ax_umap.scatter(umap_features[:, 0], umap_features[:, 1], c=colors, s=50, alpha=0.7)
    ax_umap.set_title(f'{title}', fontsize=20, weight='bold')

    # Add legend for UMAP outside the plot
    handles_umap = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_palette[i], markersize=10) for i in [0, 1]]
    labels_legend_umap = ['LS', 'SS']
    ax_umap.legend(handles_umap, labels_legend_umap, loc='upper right', bbox_to_anchor=(1, 1), fontsize=20)

    # Make x and y axis values bold
    ax_umap.tick_params(axis='both', which='major', labelsize=14, width=2)
    ax_umap.xaxis.label.set_weight('bold')
    ax_umap.yaxis.label.set_weight('bold')
    for label in ax_umap.get_xticklabels() + ax_umap.get_yticklabels():
        label.set_fontsize(14)
        label.set_weight('bold')

    # Annotate hover for UMAP
    annot_umap = ax_umap.annotate("", xy=(0,0), xytext=(20,20),
                                  textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                                  arrowprops=dict(arrowstyle="->"))
    annot_umap.set_visible(False)

    def update_annot_umap(ind):
        pos = scatter_umap.get_offsets()[ind["ind"][0]]
        annot_umap.xy = pos
        text = "{}".format(" ".join([image_names[n] for n in ind["ind"]]))
        annot_umap.set_text(text)
        annot_umap.get_bbox_patch().set_facecolor(color_palette.get(labels[ind["ind"][0]], 'gray'))
        annot_umap.get_bbox_patch().set_alpha(0.4)

    def hover_umap(event):
        vis = annot_umap.get_visible()
        if event.inaxes == ax_umap:
            cont, ind = scatter_umap.contains(event)
            if cont:
                update_annot_umap(ind)
                annot_umap.set_visible(True)
                fig_umap.canvas.draw_idle()
            else:
                if vis:
                    annot_umap.set_visible(False)
                    fig_umap.canvas.draw_idle()

    fig_umap.canvas.mpl_connect("motion_notify_event", hover_umap)
    fig_path = os.path.join(output_dir, f'{title}.png')
    plt.tight_layout()
    plt.savefig(fig_path, format='png', bbox_inches='tight')
    #plt.show()

# Perform clustering and visualize results for each model separately
for model_name, features, labels, animal_ids, image_names in [('ViT', vit_pca_features, vit_labels, vit_animal_ids, vit_image_names),
                                                               ('ResNet', resnet_pca_features, resnet_labels, resnet_animal_ids, resnet_image_names),
                                                               ('Emoti', emoti_pca_features, emoti_labels, emoti_animal_ids, emoti_image_names)]:
    # Perform PCA, t-SNE, and UMAP
    pca_features = PCA(n_components=2, random_state=42).fit_transform(features)  # Apply PCA for clustering
    tsne_features = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)  # Apply t-SNE after PCA
    umap_features = UMAP(n_components=2, random_state=42).fit_transform(features)  # Apply UMAP after PCA

    for method_name, method in clustering_methods.items():
        # Clustering on PCA-reduced features
        if method_name == 'GMM':
            pca_cluster_labels = method.fit_predict(pca_features)
            tsne_cluster_labels = method.fit_predict(tsne_features)
            umap_cluster_labels = method.fit_predict(umap_features)
        else:
            pca_cluster_labels = method.fit(pca_features).labels_
            tsne_cluster_labels = method.fit(tsne_features).labels_
            umap_cluster_labels = method.fit(umap_features).labels_

        # Plotting
        plot_clustering(pca_features, pca_cluster_labels, method_name, animal_ids, image_names, f'{model_name} + PCA + {method_name}')
        plot_clustering(tsne_features, tsne_cluster_labels, method_name, animal_ids, image_names, f'{model_name} + t-SNE + {method_name}')
        plot_clustering(umap_features, umap_cluster_labels, method_name, animal_ids, image_names, f'{model_name} + UMAP + {method_name}')

        # Optionally save results
        results_df = pd.DataFrame({
            'Image': image_names,
            'AnimalID': animal_ids,
            'Condition': labels
        })
        results_df[f'PCA_{method_name}'] = pca_cluster_labels
        results_df[f'tSNE_{method_name}'] = tsne_cluster_labels
        results_df[f'UMAP_{method_name}'] = umap_cluster_labels

        csv_filename = f'clustering_results_{model_name}_{method_name}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        results_df.to_csv(csv_path, index=False)