import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs("confusion_matrices", exist_ok=True)

# Confusion matrices (best and worst for 5 models)
conf_matrices_best = [
    np.array([[42, 0], [34, 51]]),  # ConvNeXt - Batch 1
    np.array([[42, 0], [43, 42]]),  # EfficientNet_V2 - Batch 1
    np.array([[20, 18], [9, 49]]),  # MobileNet_V3 - Batch 5
    np.array([[41, 1], [36, 49]]),  # RegNet - Batch 1
    np.array([[142, 8], [29, 114]]) # ViT - Batch 3
]

conf_matrices_worst = [
    np.array([[0, 43], [0, 133]]),       # ConvNeXt - Batch 2
    np.array([[144, 6], [133, 10]]),     # EfficientNet_V2 - Batch 3
    np.array([[143, 7], [137, 6]]),      # MobileNet_V3 - Batch 3
    np.array([[144, 6], [137, 6]]),      # RegNet - Batch 3
    np.array([[13, 30], [4, 129]])       # ViT - Batch 2
]

# Model names and batch numbers
model_names = ["ConvNeXt", "EfficientNet_V2", "MobileNet_V3", "RegNet", "ViT"]
best_batches = [1, 1, 5, 1, 3]
worst_batches = [2, 3, 3, 3, 2]

# Axis labels
labels = ["LS", "HS"]

# Function to plot and save confusion matrix
def plot_and_save_matrix(cm, model, batch, tag):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title(f"{model} - {tag} (Batch {batch})", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # Add text annotations (doubled font size for clarity)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            norm_val = norm_cm[i, j]
            text_color = 'white' if norm_val > 0.5 else 'black'
            ax.text(j, i, f'{value}', ha='center', va='center', color=text_color, fontsize=16)

    plt.tight_layout()
    filename = f"confusion_matrices/{model.replace(' ', '_')}_{tag}_Batch{batch}.jpg"
    plt.savefig(filename, dpi=300)
    plt.close()

# Loop through and plot/save all confusion matrices
for i in range(5):
    plot_and_save_matrix(conf_matrices_best[i], model_names[i], best_batches[i], "Best")
    plot_and_save_matrix(conf_matrices_worst[i], model_names[i], worst_batches[i], "Worst")
