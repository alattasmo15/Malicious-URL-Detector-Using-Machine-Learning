# Malicious-URL-Detector-Using-Machine-Learning
[Malicious URL Detector Poster.pdf](https://github.com/user-attachments/files/23155535/Malicious.URL.Detector.Poster.pdf)

Built and trained a machine learning model to classify phishing and malicious URLs from 650K+ samples using feature extraction, vectorization, and ensemble methods (Forest Tree, XGBoost, LightGBM). Evaluated with precision, recall, and F1 metrics. Presented at SUNY Albany CEHC Showcase.

This project utilizes machine learning to detect and classify dangerous URLs. The goal is to create a model that can automatically determine if a link is innocuous, phishing, malware, or defacement based on its attributes, allowing consumers and organizations to detect dangers before they propagate.

The dataset contains over 650,000 URLs, which I analyzed and classified using advanced feature extraction, visualization, and various machine learning models.

What it does:
- Extracts over 20 different URL features (such as length, IP usage, suspicious phrases, amount of dots, etc.).
- Training and comparing Random Forest, LightGBM, and XGBoost models.
- Uses progress bars to display real-time model training.
- Evaluates model performance using accuracy, precision, recall, and F1-score.

- Allows you to test your own URLs and determine whether they are safe or dangerous.

Psuedo Code: 
- Load the dataset, malicious_phish.csv.
- Extract and develop URL-based functionality.
- Encode the target labels for model training.
- Train and assess every classifier.
- Visualize metrics and feature importance.
- Test new URLs with the trained models.

Results: 
- Achieved high accuracy across all three models
- LightGBM performed best overall for both speed and consistency
- Demonstrates how ML can be applied to cyber threat detection in real-world scenarios
