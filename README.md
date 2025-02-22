# 🚀 Anomaly Detection in Mixtiles' Traffic Using Deep Learning

## 📌 Project Overview
This project enhances Mixtiles' website traffic anomaly detection using a Convolutional Neural Network (CNN)-based model inspired by DeepAnT. Traditional threshold-based methods often misclassify normal traffic fluctuations as anomalies or fail to catch actual issues. By leveraging deep learning, we aim to reduce false positives and negatives, improving Mixtiles' ability to detect meaningful deviations.

### 🔍 Key Features
- 📈 **Data-Driven Anomaly Detection**: Uses a CNN model trained on historical traffic data.
- 🏷️ **Expert-Labeled Anomalies**: Compared predictions against anomalies labeled by a domain expert.
- 📊 **Comparative Evaluation**: Benchmarks CNN performance against Mixtiles' existing model.
- 🎯 **Attention Mechanism Testing**: Explored Bahdanau Attention and Self-Attention for potential improvements.

## 📊 Results Summary
- The CNN-based model significantly outperformed Mixtiles' threshold-based method in anomaly detection.
- **F2 Score Comparison**:
  - 🏢 **Company Model**: 0.62
  - 🧠 **CNN Model**: 0.77
  - 🔍 **CNN + Attention Model**: 0.58 (Did not improve results)

### 📊 Confusion Matrix Insights:
- ✅ CNN model **reduces false positives and false negatives** compared to the company model.
- ❌ Attention models **failed to improve performance**, likely due to periodic patterns in website traffic.

## 📂 File Structure
- 📄 `README.md` – This document.
-


## 📊 Visual Results

### 📌 Time Series Predictions
The figure below showcases the model’s predicted traffic values against actual traffic counts. The CNN-based DeepAnT model successfully captures periodic trends in website visits but struggles with sharp anomalies. This is not necessarily a weakness—true anomalies are, by definition, rare and unpredictable. If the model were to predict them too accurately, it might be overfitting to noise rather than detecting meaningful deviations.

**📷 Placeholder: Insert Time Series Prediction Plot Here**

### 📊 Confusion Matrices
To assess real-world impact, we compared the CNN model’s anomaly detection with Mixtiles' existing rule-based method. The confusion matrices highlight key improvements:

- ✅ **True Positives (TP):** CNN detects **more anomalies** than the company model.
- ❌ **False Negatives (FN):** CNN reduces missed anomalies, catching more critical events.
- ⚠️ **False Positives (FP):** CNN significantly reduces unnecessary anomaly alerts.

**📷 Placeholder: Insert Confusion Matrices for CNN and Company Model Here**

## 🛠️ Attention Mechanism: Hypothesis & Results

Given the success of attention mechanisms in sequential data tasks, we tested whether they could enhance anomaly detection. The goal was to help the model focus on critical time steps rather than treating all past data equally. We implemented **Bahdanau Attention** and **Self-Attention**, following the approach outlined by:
- **Cook et al. (2020):** Surveying attention in anomaly detection for IoT time-series data.
- **Zhao et al. (2023):** Investigating attention mechanisms in time series classification.

However, results indicate that attention mechanisms **did not improve anomaly detection performance**. Website traffic follows strong **periodic trends** (daily and weekly cycles), which CNNs already capture effectively. Attention layers excel in cases where anomalies depend on long-range dependencies (e.g., fraud detection), but in this dataset, they **added unnecessary complexity**, leading to overfitting.

**📷 Placeholder: Insert Model Loss Curves for CNN and CNN + Attention Here**

## 🚀 Future Work
To further improve anomaly detection, future research can explore:

- **Anomaly-Aware Training**: Exclude anomalies during training to improve robustness.
- **External Features**: Incorporate marketing campaigns, server issues, and promotions as inputs.
- **Hybrid Models**: Combine CNNs with LSTMs or Transformers for improved long-range pattern recognition.
- **Threshold Optimization**: Fine-tune anomaly detection thresholds to better balance precision and recall.
- **Data Augmentation**: Generate synthetic anomalies to improve model generalization.

---

This study demonstrates that a **CNN-based DeepAnT model significantly outperforms Mixtiles' existing threshold-based method**, reducing false negatives and false positives. While attention mechanisms did not yield expected improvements, refining model architecture and feature engineering could further enhance detection accuracy.

---

## 🔗 References
(References remain unchanged from the previous section)

---

### 📌 How to Cite This Work
If you use this repository, please cite:
