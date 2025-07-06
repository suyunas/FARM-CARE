import numpy as np
import matplotlib.pyplot as plt

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

# Corresponding model names and batch numbers
model_names = ["ConvNeXt", "EfficientNet_V2", "MobileNet_V3", "RegNet", "Vision Transformer (ViT)"]
best_batches = [1, 1, 5, 1, 3]
worst_batches = [2, 3, 3, 3, 2]

# Labels
labels = ["Low Stressed (LS)", "High Stressed (LS)"]

# Plotting
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Row-Normalized Confusion Matrices for Best and Worst Model Batches", fontsize=16)

def plot_matrix(cm, ax, title):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm, row_sums, where=row_sums != 0)

    im = ax.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            norm_val = norm_cm[i, j]
            text_color = 'white' if norm_val > 0.5 else 'black'
            ax.text(j, i, f'{value}', ha='center', va='center', color=text_color)

# Plot best and worst matrices with model names and batch numbers
for i in range(5):
    plot_matrix(conf_matrices_best[i], axes[0, i], f"{model_names[i]} - Best (Batch {best_batches[i]})")
    plot_matrix(conf_matrices_worst[i], axes[1, i], f"{model_names[i]} - Worst (Batch {worst_batches[i]})")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()