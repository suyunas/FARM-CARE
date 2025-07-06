import numpy as np
import matplotlib.pyplot as plt
import os

# Confusion matrices (best and worst for 5 models)
conf_matrices_best = [
    (np.array([[42, 0], [34, 51]]), "ConvNeXt", "Batch1"),
    (np.array([[42, 0], [43, 42]]), "EfficientNet_V2", "Batch1"),
    (np.array([[20, 18], [9, 49]]), "MobileNet_V3", "Batch4"),
    (np.array([[41, 1], [36, 49]]), "RegNet", "Batch1"),
    (np.array([[142, 8], [29, 114]]), "ViT", "Batch3")
]

conf_matrices_worst = [
    (np.array([[0, 43], [0, 133]]), "ConvNeXt", "Batch2"),
    (np.array([[144, 6], [133, 10]]), "EfficientNet_V2", "Batch3"),
    (np.array([[143, 7], [137, 6]]), "MobileNet_V3", "Batch3"),
    (np.array([[144, 6], [137, 6]]), "RegNet", "Batch3"),
    (np.array([[13, 30], [4, 129]]), "ViT", "Batch2")
]

labels = ["Unstressed", "Stressed"]  # More standard terminology

# Directory to save confusion matrix images
output_dir = r'J:\old_results\gradcam'
os.makedirs(output_dir, exist_ok=True)

def save_conf_matrix(cm, model, batch, type_label):
    fig, ax = plt.subplots(figsize=(5, 4))

    # Normalize the confusion matrix row-wise
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm, row_sums, where=row_sums != 0)

    ax.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title(f"{model} - {batch}", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Add raw count annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            norm_val = norm_cm[i, j]
            color = 'white' if norm_val > 0.5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', color=color)

    # Save the figure
    filename = f"{type_label}_{model}_{batch}.png".replace(" ", "_").lower()
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# Generate and save all matrices
for cm, model, batch in conf_matrices_best:
    save_conf_matrix(cm, model, batch, "best")

for cm, model, batch in conf_matrices_worst:
    save_conf_matrix(cm, model, batch, "worst")

print(f"Confusion matrices saved to: {output_dir}")