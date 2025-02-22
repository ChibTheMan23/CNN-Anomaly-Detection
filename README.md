# CNN-Anomaly-Detection
# Anomaly Detection in Mixtiles' Traffic Using Deep Learning

## Project Overview
This project aims to improve Mixtiles' website traffic anomaly detection using a Convolutional Neural Network (CNN)-based model inspired by DeepAnT. Traditional threshold-based methods often misclassify normal traffic fluctuations as anomalies or fail to catch actual issues. By leveraging deep learning, we aim to reduce false positives and negatives, enhancing Mixtiles' ability to detect meaningful deviations.

### Key Features
- **Data-Driven Anomaly Detection**: Uses a CNN model trained on historical traffic data.
- **Expert-Labeled Anomalies**: Compared predictions against anomalies labeled by a domain expert.
- **Comparative Evaluation**: Benchmarks CNN performance against Mixtiles' existing model.
- **Attention Mechanism Testing**: Explored Bahdanau Attention and Self-Attention for potential improvements.

## Results Summary
- The CNN-based model significantly outperformed Mixtiles' threshold-based method in anomaly detection.
- **F2 Score Comparison**:
  - **Company Model**: 0.62
  - **CNN Model**: 0.77
  - **CNN + Attention Model**: 0.58 (Did not improve results)
- **Confusion Matrix Insights**:
  - CNN model reduces false positives and false negatives compared to the company model.
  - Attention models failed to improve performance, likely due to periodic patterns in website traffic.

## File Structure
- `README.md` – This document.
- `anomaly_detection_notebook.ipynb` – Jupyter Notebook with full implementation and analysis.
- `figures/` – Contains key figures and result visualizations.

## References
1. **DeepAnT Framework**:
   - Malhotra, Pankaj, et al. "LSTM-based encoder-decoder for multi-sensor anomaly detection." arXiv preprint arXiv:1607.00148 (2016).
2. **Attention Mechanism in Anomaly Detection**:
   - Cook, A. A., Misirli, G., & Fan, Z. (2020). "Anomaly detection for IoT time-series data: A survey." *IEEE Internet of Things Journal, 7(7),* 6481–6494.
   - Zhao, B., Xing, H., Wang, X., Song, F., & Xiao, Z. (2023). "Rethinking attention mechanism in time series classification."
3. **Time-Series Forecasting with Deep Learning**:
   - Zheng, A., & Casari, A. (2018). "Feature engineering for machine learning: Principles and techniques for data scientists." O'Reilly Media.

