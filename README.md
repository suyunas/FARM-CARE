# Farm Care: Deep Learning for Robust Stress Classification in Sows

**Farm Care** is an AI-based framework for **automated, non-invasive stress detection** in sows using facial images and deep learning  
This repository includes code, data samples, notebooks, and documentation for:

- **Study 1:** Stress classification within one generation
- **Study 2:** Cross-generation evaluation (training on parents, testing on offspring)

---

## Project Motivation

Livestock stress impacts animal welfare, productivity, and farm economics
Traditional stress assessment methods are manual, subjective, and time-consuming
**Farm Care** uses facial region analysis and deep learning to enable reliable, repeatable, and real-time stress classification in sows

---

## Methodology

**Data Collection**  
- Images of sows from multiple cameras covering different angles

**Preprocessing**  
- Image Labelling using **Segment Anything**
- Automatic facial detection and cropping using **YOLOv8**
- Pig Eye-Based Image Filtering using **YOLOv3-Tiny**
- Redundant Image filter using **VGG16**
- Image augmentation and filtering to improve data quality

![image](https://github.com/user-attachments/assets/929762a8-a427-4451-ae69-dfc3a5cbe796)


**Models**  
- Pre-trained Deeep Learning Models: **ConvNeXt, EfficientNetV2, MobileNetV3, RegNet, and Vision Transformer (ViT)**
- Fine-tuned for stress classification

**Experiments**  
- Within-generation: Train/test on same generation (12 sows)
- Cross-generation: Train on 12 parent sows, test on their 48 offspring to assess model generalization

**Performance Metrics**  
- Accuracy, precision, recall, confusion matrix

---

## Key Results

- **High classification accuracy** with Vision Transformers (ViT)
- Robust detection with various data augmentation setting
- Promising cross-generation results, showing potential for practical deployment
  
![image](https://github.com/user-attachments/assets/dee21982-0850-4a70-acb3-f9e3f3dbc0ba)

## Conclusion

The Vision Transformer model (ViT) emerged as the most effective architecture for classifying stress states in sows, consistently outperforming other models across multiple evaluation batches. It achieved the highest average accuracy of 0.75, with an F1-score of 0.78 overall, and performed particularly well in identifying high-stressed sows—achieving up to 0.88 F1-score in its best-performing batch

---

## Getting Started

### Clone the Repo

```bash
git clone https://github.com/YOUR-USERNAME/FARM-CARE.git
cd FARM-CARE
