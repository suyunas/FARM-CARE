# FARM interventions to Control Antimicrobial ResistancE (FARM-CARE)

In this project we have utilized an AI-based framework for **automated, non-invasive stress detection** in sows using facial images and deep learning  
This repository includes code, data samples, notebooks, and documentation for:

- **Study 1:** Stress classification within one generation
- **Study 2:** Cross-generation evaluation (training on parents, testing on offsprings)

---

## Project Motivation

Livestock stress impacts animal welfare, productivity, and farm economics
Traditional stress assessment methods are manual, subjective, and time-consuming
Our method uses facial region analysis and deep learning to enable reliable, repeatable, and real-time stress classification in sows

---

## Methodology

**Data Collection**  
- Images of sows from multiple cameras covering different angles

![image](https://github.com/user-attachments/assets/929762a8-a427-4451-ae69-dfc3a5cbe796)
Figure 1. Simultaneous image capturing from 4 sows (left), full view from individual stall (middletop),
image cropping and bar removal from full view (middle-bottom) and segmented faces (right).

**Preprocessing**  
- Image Labelling using **Segment Anything**
- Automatic facial detection and cropping using **YOLOv8**
- Pig Eye-Based Image Filtering using **YOLOv3-Tiny**
- Redundant Image filter using **VGG16**

![image](https://github.com/user-attachments/assets/5c6fd238-8fc3-4f14-816b-b65af7874a5c)
Figure 2. Data collection and processing pipeline.

**Models** 
- Image augmentation and filtering during training to improve data quality
- Pre-trained Deeep Learning Models: **ConvNeXt, EfficientNetV2, MobileNetV3, RegNet, and Vision Transformer (ViT)**
- Fine-tuned for stress classification

**Experiments**  
- Within-generation: Train/test on same generation (12 sows) leave one out cross validation (LOCO) approach
- Cross-generation: Train on 12 parent sows, test on their 48 offspring to assess model generalization

**Performance Metrics**  
- Accuracy, precision, recall, confusion matrix

---

## Key Results

- **High classification accuracy** with Vision Transformers (ViT)
- Robust detection with various data augmentation setting
- Promising cross-generation results, showing potential for practical deployment
  
![image](https://github.com/user-attachments/assets/dee21982-0850-4a70-acb3-f9e3f3dbc0ba)

Table 1. Model predictions for Low-Stressed (LS) and High-Stressed (HS) sow images across different
deep learning architectures.

## Conclusion

The Vision Transformer model (ViT) emerged as the most effective architecture for classifying stress states in sows, consistently outperforming other models across multiple evaluation batches. It achieved the highest average accuracy of 0.75, with an F1-score of 0.78 overall, and performed particularly well in identifying high-stressed sows—achieving up to 0.88 F1-score in its best-performing batch.

---

## Getting Started

Follow these steps to run the training and testing workflows:

### Clone the Repository

git clone https://github.com/YOUR-USERNAME/FARM-CARE.git

cd FARM-CARE

### Prepare Input Data

- Your input images should be listed in **CSV or Excel** files
- The **first column** must contain the **Path**(image path)
- The **second column** must contain the **Condition**(corresponding label 0/1)

### Run Training

1. Navigate to the training directory:

   cd Training_Parents
3. Run the training script:

   python Train_Parent_Data.py

### Run Testing

1. Arrange your test images in the same **CSV or Excel** format.
2. Navigate to the test directory:

   cd ../Testing Parents
4. Run the testing script:

   python Test_Parent_Data.py

## Notes

Make sure your image paths in your CSV/Excel files are correct and relative to your project folder.  
Use a virtual environment and install all dependencies with:

pip install -r requirements.txt

##  Example Project Structure

FARM-CARE/

├── training_Parents/

│   └── training_data.py

├── test_Parents/

│   └── test_parent.py

├── data/

│   ├── train.csv  # or train.xlsx

│   ├── test.csv   # or test.xlsx

├── images/

│   ├── sow1.jpg

│   ├── sow2.jpg

│   └── ...

├── requirements.txt

├── README.md

└── ...

---

## Funding

This research was funded by the Joint Programming Initiative on Antimicrobial Resistance (JPIAMR) under the FARM-CARE project, ‘FARM interventions to Control Antimicrobial ResistancE – Full Stage’ (Project ID: 7429446), and by the Medical Research Council (MRC), UK (Grant Number: MR/W031264/1).

## Institutional Review Board Statement
This study underwent internal ethical review by both SRUC’s and UWE Bristol’s AnimalWelfare and Ethical Review Bodies (ED AE 16-2019 and R101) and was carried out under the UK Home Office license (P3850A80D).

---


