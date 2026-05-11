# Exercise-3: XGBoost for Medical Image Classification

## Overview

This project focuses on implementing and comparing different XGBoost-based approaches for medical image classification using:

* Chest X-Ray Dataset
* Brain Tumor MRI Dataset

The following methods were implemented and evaluated:

1. Original XGBoost
2. PCA + XGBoost
3. VGG16 + XGBoost
4. Comparison with CNN and SVM approaches

The project also analyzes the effect of changing the training data size using:

* 20% of training data
* 40% of training data
* 60% of training data
* 80% of training data

The performance trend of all methods is studied using accuracy plots.



---

# Algorithms Used

## 1. Original XGBoost

XGBoost is a gradient boosting algorithm that combines multiple weak decision tree learners to improve prediction accuracy.

### Workflow

* Load image dataset
* Resize and preprocess images
* Flatten image pixels into feature vectors
* Train XGBoost classifier
* Evaluate using accuracy, precision, recall, and F1-score

---

## 2. PCA + XGBoost

Principal Component Analysis (PCA) is used for dimensionality reduction before training the XGBoost classifier.

### Workflow

* Extract image features
* Apply PCA to reduce dimensions
* Train XGBoost on reduced features
* Evaluate performance

### Advantage

* Reduces computational complexity
* Removes redundant features
* Improves training speed

---

## 3. VGG16 + XGBoost

VGG16 pretrained CNN is used as a feature extractor.

### Workflow

* Load pretrained VGG16 model
* Extract deep features from images
* Feed extracted features into XGBoost classifier
* Evaluate performance

### Advantage

* Deep feature extraction improves classification quality
* Better representation learning compared to raw pixels

---



# Experimental Results

# 1. VGG16 + XGBoost (Chest X-Ray)

| Metric    | Train  | Validation | Test   |
| --------- | ------ | ---------- | ------ |
| Accuracy  | 1.0000 | 0.9375     | 0.7644 |
| Precision | 1.0000 | 0.9444     | 0.8217 |
| Recall    | 1.0000 | 0.9375     | 0.7644 |
| F1 Score  | 1.0000 | 0.9373     | 0.7308 |

### Training Data Analysis

| Training Data | Accuracy |
| ------------- | -------- |
| 20%           | 74.04%   |
| 40%           | 76.92%   |
| 60%           | 76.76%   |
| 80%           | 77.88%   |

Accuracy List:

```python
[74.03, 76.92, 76.76, 77.88]
```
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/c71af849-047f-487a-b9ba-98f75341b651" />
---

# 2. VGG16 + XGBoost (Brain MRI)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.6675 |
| Precision | 0.6815 |
| Recall    | 0.6675 |
| F1 Score  | 0.6348 |

### Training Data Analysis

| Training Data | Accuracy |
| ------------- | -------- |
| 20%           | 51.02%   |
| 40%           | 56.35%   |
| 60%           | 59.90%   |
| 80%           | 62.18%   |



<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/f8a11164-297e-4b5f-9dd4-87a3aecea853" />
---

# 3. PCA + XGBoost (Chest X-Ray)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 76.44% |
| Precision | 81.53% |
| Recall    | 76.44% |
| F1 Score  | 73.25% |

### Training Data Analysis

| Training Data | Accuracy |
| ------------- | -------- |
| 20%           | 75.16%   |
| 40%           | 74.36%   |
| 60%           | 75.00%   |
| 80%           | 75.80%   |

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/d817f696-5953-4b3a-bac2-026f5218f6bb" />

---

# 4. Original XGBoost (Chest X-Ray)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 72.92% |
| Precision | 78.83% |
| Recall    | 72.92% |
| F1 Score  | 68.24% |

### Training Data Analysis

| Training Data | Accuracy |
| ------------- | -------- |
| 20%           | 70.35%   |
| 40%           | 71.96%   |
| 60%           | 73.08%   |
| 80%           | 72.28%   |
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/9d3181d7-5a56-4033-9611-5655329a2fa7" />

---

# 5. Original XGBoost (Brain MRI)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 52.79% |
| Precision | 61.88% |
| Recall    | 52.79% |
| F1 Score  | 49.46% |

### Training Data Analysis

| Training Data | Accuracy        |
| ------------- | --------------- |
| 20%           | 47.46%          |
| 40%           | 43.65%          |
| 60%           | 51.27%          |
| 80%           | Add your result |
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/5693ef17-11c2-4245-ad61-aec1951677d3" />

---

# 6. PCA + XGBoost (Brain MRI)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 59.90% |
| Precision | 59.55% |
| Recall    | 59.90% |
| F1 Score  | 57.10% |

### Training Data Analysis

| Training Data | Accuracy |
| ------------- | -------- |
| 20%           | 43.65%   |
| 40%           | 48.73%   |
| 60%           | 55.33%   |
| 80%           | 56.35%   |
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/f4ca1f38-5af9-4c10-b3b9-f6ea06cbd35b" />

---

# 7. SVM Results

| Training Data | RBF Accuracy | Linear Accuracy |
| ------------- | ------------ | --------------- |
| 20%           | 0.7628       | 0.7452          |
| 40%           | 0.7596       | 0.7548          |
| 60%           | 0.7596       | 0.7516          |
| 80%           | 0.7564       | 0.7436          |

### Observation

SVM performance remained relatively stable across different training dataset sizes. The improvement in accuracy was limited compared to deep feature extraction methods such as VGG16 + XGBoost.

---

# 8. CNN Results

| Training Data | Accuracy | F1-score | Sensitivity | Specificity |
| ------------- | -------- | -------- | ----------- | ----------- |
| 20%           | 0.3125   | 0.2667   | 0.25        | 0.375       |
| 40%           | 0.5625   | 0.5882   | 0.625       | 0.5         |
| 60%           | 0.5000   | 0.5556   | 0.625       | 0.375       |
| 80%           | 0.6250   | 0.6667   | 0.75        | 0.5         |

### Observation

CNN performance improved as the amount of training data increased. However, the model required significantly more data to achieve stable and higher accuracy.

---

# Accuracy Comparison

## Observations

* VGG16 + XGBoost achieved the best overall performance because deep CNN features provide better image representation.
* PCA + XGBoost performed better than original XGBoost due to dimensionality reduction and removal of redundant features.
* Original XGBoost showed lower performance because raw image pixels are less informative compared to extracted deep features.
* Accuracy generally improved as the amount of training data increased.
* Larger datasets help the models learn more generalized patterns and reduce overfitting.

---





---



# Conclusion

This project demonstrates how feature extraction and dimensionality reduction significantly affect image classification performance.

Among all methods:

* VGG16 + XGBoost achieved the best performance.
* PCA + XGBoost improved efficiency while maintaining good accuracy.
* Original XGBoost performed comparatively lower due to limited feature representation.

Increasing the amount of training data generally improved model accuracy across all approaches.

---


