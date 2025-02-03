# Machine-Learning: Tree Based Classification with Imbalanced Data

## 🔷 Project Description
This project focuses on **tree-based classification methods** applied to the **APS Failure dataset**, a highly imbalanced dataset. The study explores **Random Forests, XGBoost, and Model Trees**, incorporating **SMOTE (Synthetic Minority Over-sampling Technique)** for class balance correction. The performance is evaluated using metrics such as **confusion matrix, ROC, AUC, and out-of-bag error estimation**.

---

## 🔷 Dataset Description
The APS failure dataset, sourced from the from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks), consists of:
- **60,000 training samples** (1,000 positive cases, highly imbalanced).
- **171 total columns**, including one target class column.
- **All attributes are numerical**.
- **Contains missing values** that require appropriate handling.

---

## 🔷 Libraries Used
- **Python Libraries**:
  - `pandas`, `numpy` - Data manipulation & preprocessing
  - `matplotlib`, `seaborn` - Data visualization
  - `scikit-learn` - Machine learning models & evaluation
  - `xgboost` - Extreme Gradient Boosting implementation
  - `imblearn` - Implements **SMOTE** for handling imbalanced datasets

---

## 🔷 Steps Taken to Accomplish the Project

### 🔶 1. Data Preprocessing and Exploratory Data Analysis (EDA)
- **Handled missing values** using imputation techniques.
- **Computed coefficient of variation (CV)** for all 170 features to assess feature importance.
- **Plotted a correlation matrix** to examine feature relationships.
- **Visualized top features** (with highest CV) using scatter plots and box plots.
- **Checked for class imbalance** to confirm the need for resampling.

### 🔶 2. Training a Random Forest Classifier
- Trained a **random forest model** on the dataset **without handling class imbalance**.
- Evaluated performance using:
  - **Confusion Matrix**
  - **ROC (Receiver Operating Characteristic) Curve**
  - **AUC (Area Under Curve)**
  - **Misclassification rate**
- **Calculated Out-of-Bag (OOB) error** and compared it with the test error.

### 🔶 3. Addressing Class Imbalance in Random Forests
- **Resampled the dataset** to handle class imbalance.
- **Re-trained the Random Forest model** with compensation for class imbalance.
- **Compared results** before and after resampling.

### 🔶 4. Training XGBoost and Model Trees
- **Implemented XGBoost** with **L1-penalized logistic regression** at each decision node.
- **Performed cross-validation** (5-fold, 10-fold, and leave-one-out methods) to tune the **regularization parameter (α)**.
- **Compared test error with cross-validation estimates**.
- **Reported Confusion Matrix, ROC, and AUC** for the training and test sets.

### 🔶 5. Implementing SMOTE for Data Balancing
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class.
- **Re-trained XGBoost** with L1-regularized logistic regression after SMOTE.
- **Repeated cross-validation and performance analysis**.
- **Compared results of the SMOTE-adjusted model vs. the original model**.

---
## 📌 **Note**
This repository contains a **Jupyter Notebook** detailing each step, along with **results and visualizations**.
