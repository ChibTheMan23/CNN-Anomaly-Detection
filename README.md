# Anomaly Detection in Website Traffic Using Deep Learning

## 📌 Project Overview
This project enhances website traffic anomaly detection using DeepAnT, a Convolutional Neural Network (CNN)-based approach designed for time series anomaly detection. DeepAnT operates by predicting future values in a time series and identifying anomalies when significant deviations occur between the predicted and actual values. Traditional threshold-based methods often misclassify normal traffic fluctuations as anomalies or fail to catch actual issues. By leveraging deep learning, I aim to reduce false positives and false negatives, enhancing the ability to detect meaningful deviations.

### 🔍 Key Features
- DeepAnT-Based Detection: Uses a CNN model trained on historical traffic data to predict future values.
- Anomaly Detection Mechanism: Flags anomalies based on deviations between predicted and actual values.
- Expert-Labeled Anomalies: Compared predictions against anomalies labeled by a domain expert.
- Comparative Evaluation: Benchmarks CNN performance against the existing model.
- Attention Mechanism Testing: Explored Bahdanau Attention and Self-Attention for potential improvements.

## 📈 Results Summary
- The CNN-based model significantly outperformed the threshold-based method in anomaly detection.
- **F2 Score Comparison**:
  - **Company Model**: 0.29
  - **CNN Model**: 0.62
  - **CNN + Attention Model**: 0.64

- CNN model reduces false positives and false negatives compared to the company model.
- Attention models provided slight improvements but did not dramatically enhance performance.

## 📊 Visual Results

### Time Series Predictions
The figure below showcases the model’s predicted traffic values against actual traffic counts. The CNN-based DeepAnT model successfully captures periodic trends in website visits but struggles with sharp anomalies. This is not necessarily a weakness—true anomalies are, by definition, rare and unpredictable. If the model were to predict them too accurately, it might be overfitting to noise rather than detecting meaningful deviations.


![tesx](https://github.com/ChibTheMan23/CNN-Anomaly-Detection/blob/94b267cd4eac5a9cdf2c84f184fadb94b01ba14a/figures/Deep-AnT%20Predictions.png)

### Confusion Matrices
To assess real-world impact, I compared the CNN model’s anomaly detection with the existing rule-based method. The confusion matrices highlight key improvements:

- **True Positives (TP):** CNN detects **more anomalies** than the company model.
- **False Negatives (FN):** CNN reduces missed anomalies, catching more critical events.
- **False Positives (FP):** CNN significantly reduces unnecessary anomaly alerts.

![tesx](https://github.com/ChibTheMan23/CNN-Anomaly-Detection/blob/f424d99ba85128c98c38cd8cfc9758e1653afb8e/figures/Confusion%20Figure.png)


## 🛠️ Attention Mechanism: Hypothesis & Results

Given the success of attention mechanisms in sequential data tasks, I tested whether they could enhance anomaly detection. The goal was to help the model focus on critical time steps rather than treating all past data equally. I implemented Bahdanau Attention.
Results indicate that attention mechanisms slightly improved anomaly detection performance. Website traffic follows strong periodic trends (daily and weekly cycles), which CNNs already capture effectively. Attention layers helped refine predictions, particularly in detecting traffic spikes and drops. However, the overall performance gains were modest, suggesting that CNNs alone are already well-suited for capturing short-term dependencies in this dataset.

![Loss](https://github.com/ChibTheMan23/CNN-Anomaly-Detection/blob/9cbf3c53163a63f2f470b4a924887c8b92871f43/figures/Loss%20Plot.png)

## 🚀 Future Work
To further improve anomaly detection, future research can explore:

- **Anomaly-Aware Training**: Exclude anomalies during training to improve robustness.
- **Threshold Optimization**: Fine-tune anomaly detection thresholds to better balance precision and recall.
- **Data Augmentation**: Generate synthetic anomalies to improve model generalization.

---

This study demonstrates that a **CNN-based DeepAnT model significantly outperforms the existing threshold-based method**, reducing false negatives and false positives. While attention mechanisms did not yield expected improvements, refining model architecture and feature engineering could further enhance detection accuracy.

---

## 🔗 References
1. M. Munir, S. A. Siddiqui, A. Dengel and S. Ahmed, "DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series," in IEEE Access, vol. 7, pp. 1991-2005, 2019, doi: 10.1109/ACCESS.2018.2886457.
2. Andrew A. Cook, Goksel Misirli, and Zhong Fan. Anomaly detection for IoT time-series data: A survey. IEEE Internet of Things Journal, 7(7):6481–6494, 2020.
3. Bowen Zhao, Huanlai Xing, Xinhan Wang, Fuhong Song, and Zhiwen Xiao. Rethinking attention mechanism in time series classification. Retrieved from: https://arxiv.org/abs/2207.07564.

