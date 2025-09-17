# Credit-Card-Fraud-Detection
This project develops a deep learning model to detect fraudulent transactions in financial datasets. Fraud detection is critical in banking, e-commerce, and payment systems, where early identification of suspicious activity helps reduce financial losses and improve security.

Project Overview

Objective: Build and evaluate a deep learning model for classifying transactions as fraudulent or legitimate.

Dataset: Transaction records with features such as amount, type, timestamp, and anonymized identifiers.

Approach:

Data cleaning and preprocessing (handling imbalance, scaling, encoding).

Exploratory Data Analysis (EDA) with visualizations.

Building a neural network classifier with TensorFlow/Keras.

Training, validation, and evaluation using metrics like accuracy, precision, recall, and F1-score.

Features

Handles class imbalance using oversampling/undersampling or SMOTE.

Includes correlation heatmaps and feature analysis.

Deep learning architecture with multiple hidden layers and dropout for regularization.

Performance evaluation with confusion matrix and ROC-AUC curve.

Technologies Used

Python

TensorFlow / Keras

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Results

The model achieved strong fraud detection capability, with balanced precision and recall, ensuring both detection accuracy and minimal false alarms.

How to Run

Clone the repository:

git clone https://github.com/morgakhweku/Credit-Card-Fraud-Detection.git


Install dependencies:

pip install -r requirements.txt


Run the notebook or script to train the model and evaluate results.

Future Work

Experiment with advanced architectures (LSTMs, Transformers).

Deploy model via Flask/FastAPI for real-time fraud detection.

Integrate with cloud platforms for large-scale transaction monitoring.
