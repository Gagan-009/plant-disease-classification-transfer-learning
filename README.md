\# Plant Disease Classification Using Transfer Learning



Comparative analysis of transfer learning models for plant disease classification.



\## Overview



This project investigates the use of pretrained convolutional neural networks for multiclass plant disease classification using the PlantVillage dataset.



The project compares three pretrained architectures:



\- ResNet18

\- ResNet50

\- EfficientNet-B0



For each architecture, two strategies will be evaluated:



1\. Feature extraction with the pretrained backbone frozen

2\. Fine-tuning with selected backbone layers unfrozen



A small CNN trained from scratch will also be used as a baseline.



\## Project Pipeline



PlantVillage Dataset

→ Data Preparation

→ Stratified Train/Validation/Test Split

→ Baseline CNN

→ Feature Extraction

→ Fine-Tuning

→ Evaluation

→ Error Analysis

→ Grad-CAM

→ Streamlit Demonstration



\## Evaluation



The models will be compared using:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Macro-F1

\- Confusion matrix

\- Parameter count

\- Training time

\- Inference time



The final analysis will consider both predictive performance and computational cost.



\## Repository Structure



```text

data/

├── raw/

├── processed/

└── splits/



notebooks/

src/

models/

results/

├── metrics/

└── figures/

app/

report/

