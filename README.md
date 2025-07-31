# 💳 Fraudulent Transaction Detection using Ensemble Learning

## 📚 Abstract

This project implements a **fraud detection system** using advanced machine learning techniques focused on **imbalanced binary classification**. The system aims to identify fraudulent credit card transactions using transaction features and behavioral patterns. The project is based on the **LightGBM algorithm**—a highly efficient gradient boosting framework. Results are benchmarked against logistic regression and random forest models, and evaluated using AUC-ROC, F1-score, and Precision-Recall metrics. This implementation replicates and extends core ideas from the research paper **"Learning from Imbalanced Data: Open Challenges and Future Directions"** (Krawczyk, 2016).

---

## 🧠 Problem Statement

Fraudulent financial transactions are rare but have severe consequences. Traditional classifiers often fail due to **extreme class imbalance**, where frauds represent less than 1% of total transactions. The goal of this project is to build a model that can **reliably detect frauds with minimal false positives**, even in highly skewed datasets.

---

## 💡 Solution Overview

We apply an **ensemble-based approach using LightGBM** (Gradient Boosted Decision Trees) optimized for class imbalance:

### 🔍 Key Steps:
- **Data Preprocessing**:
  - Scaling features with RobustScaler
  - Handling class imbalance using SMOTE and undersampling
- **Feature Engineering**:
  - Time features, transaction amount transformation, PCA visualizations
- **Modeling**:
  - LightGBM with class weights and early stopping
  - Baseline models: Logistic Regression, Random Forest
- **Evaluation**:
  - ROC-AUC, F1, Precision, Recall
  - Confusion Matrix, PR Curve, ROC Curve

---

## 🤖 Algorithm: LightGBM (Gradient Boosted Decision Trees)

- Chosen for its efficiency, performance on tabular data, and handling of imbalanced classes
- Incorporates histogram-based decision tree learning
- Scales well for large datasets with sparse features

---

## 📊 Results Summary

| Model              | ROC-AUC | F1-Score | Precision@Fraud |
|-------------------|---------|----------|------------------|
| Logistic Regression | 0.91    | 0.62     | 0.84             |
| Random Forest       | 0.95    | 0.71     | 0.88             |
| **LightGBM (final)**     | **0.98** | **0.79**  | **0.92**          |

- LightGBM significantly outperformed other models.
- Achieved high precision with low false positive rate.
- Final model generalized well to unseen fraud types.
- 
## Distribution of Transaction Type(Screenshot):
![Distribution](https://github.com/user-attachments/assets/7b08bd49-27e6-49bf-afec-a6077eaa325e

---


## 🧪 Dataset

- **Kaggle Credit Card Fraud Dataset**: [Link](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions with only 492 labeled as fraud
- Feature anonymized (V1–V28), `Amount`, and `Time`

---

## 🔬 Research Basis

**Reference Paper**:
> Krawczyk, B. (2016). *Learning from Imbalanced Data: Open Challenges and Future Directions*. Progress in Artificial Intelligence.  
> [Paper Link](https://link.springer.com/article/10.1007/s13748-016-0094-0)

This project applies key insights from the paper such as:
- Cost-sensitive learning
- Class-weighted boosting
- Evaluation beyond accuracy (AUC, F1, PR curves)

---

## 🧠 Research Potential

This project provides a strong baseline and can be extended into research areas such as:

- **Explainable AI (XAI)** for fraud detection (e.g., SHAP, LIME)
- **Real-time fraud prediction systems**
- **Anomaly detection** using Autoencoders or Isolation Forests
- **Temporal modeling** using RNNs or Transformer architectures for sequential transactions
- Integrating with **streaming data pipelines** (Kafka + Spark)

---

## ⚙️ Technologies

- Python 3.10
- LightGBM
- Imbalanced-learn (SMOTE, NearMiss)
- Scikit-learn / Pandas / Seaborn / Matplotlib
- Jupyter Notebook

---

📄 References
Krawczyk, B. Learning from Imbalanced Data, Springer (2016)

Kaggle Dataset: Credit Card Fraud Detection

LightGBM Docs: https://lightgbm.readthedocs.io/

