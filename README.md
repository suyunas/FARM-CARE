# 🐷 Farm Care: Deep Learning for Robust Stress Classification in Sows

**Farm Care** is an AI-based framework for **automated, non-invasive stress detection** in sows using facial images and deep learning.  
This repository includes code, data samples, notebooks, and documentation for:

- 📄 **Study 1:** Stress classification within one generation (published)
- 🧬 **Study 2:** Cross-generation evaluation (training on parents, testing on offspring)

---

## 📚 Project Motivation

Livestock stress impacts animal welfare, productivity, and farm economics.  
Traditional stress assessment methods are manual, subjective, and time-consuming.  
**Farm Care** uses facial region analysis and deep learning to enable reliable, repeatable, and real-time stress classification in sows.

---

## 🧩 Methodology

✅ **Data Collection**  
- Images of sows from multiple cameras covering different angles.
- Region-of-interest (ROI) detection using **YOLOv8**.

✅ **Preprocessing**  
- Automatic facial detection and cropping.
- Image augmentation and filtering to improve data quality.

✅ **Models**  
- Pre-trained CNN architectures: ResNet18, ResNet50, ConvNeXt, Vision Transformers (ViT).
- Fine-tuned for stress classification.

✅ **Experiments**  
- Within-generation: Train/test on same generation (24 sows).
- Cross-generation: Train on parent sows, test on their offspring to assess model generalization.

✅ **Performance Metrics**  
- Accuracy, precision, recall, confusion matrix.
- Ablation studies to test robustness.

---

## 🏆 Key Results

- **High classification accuracy** with YOLOv8 and deep CNN models.
- Robust detection with various data augmentation settings.
- Promising cross-generation results, showing potential for practical deployment.

---

## 🚀 Getting Started

### 📦 Clone the Repo

```bash
git clone https://github.com/YOUR-USERNAME/FARM-CARE.git
cd FARM-CARE
