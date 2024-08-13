# Credit Card Fraud Detection

Welcome to my exploration of credit card fraud detection! This project involves exploring transaction data to identify patterns of fraudulent activities and implementing various machine learning models to predict fraud.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Objective](#objective)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, I explore transaction data to understand patterns of credit card fraud. The goal is to build and evaluate machine learning models that can accurately predict fraudulent transactions. The project implements techniques such as SMOTE for handling imbalanced data and various machine learning algorithms, including supervised and semi-supervised models, as well as a deep learning model.

## Dataset

The dataset contains credit card transactions made by European cardholders in September 2013. It includes transactions over two days, with 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

- **Features:**
  - The dataset contains numerical features resulting from a PCA transformation. 
  - Features V1, V2, ..., V28 are principal components.
  - The 'Time' feature indicates the seconds elapsed between transactions.
  - The 'Amount' feature represents the transaction amount.
  - The 'Class' feature is the response variable, where 1 indicates fraud and 0 indicates a normal transaction.

The dataset was collected and analyzed during a research collaboration between Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles).

## Objective

The main objective is to explore the dataset and implement machine learning models to predict fraudulent transactions. The project aims to serve as a practical study of credit card fraud detection using machine learning techniques.

## Methodology

1. **Data Exploration:** 
   - Analyze the distribution of data, focusing on numerical features and class imbalance.
   
2. **Data Preprocessing:**
   - Apply SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
   
3. **Model Implementation:**
   - Implement various machine learning models, including:
     - Random Forest
     - Semi-supervised learning algorithms
     - Deep learning models

4. **Model Evaluation:**
   - Evaluate model performance using metrics like the Area Under the Precision-Recall Curve (AUPRC), given the class imbalance.

## Results

- The models were assessed based on their ability to accurately predict fraud while minimizing false positives and false negatives.
- A summary of key findings and performance metrics for each model is included in the notebook.

## Conclusion

This project demonstrates the application of various machine learning techniques to address the challenge of fraud detection in credit card transactions. The use of SMOTE and evaluation metrics like AUPRC is crucial in handling the class imbalance and assessing model performance effectively.

## Installation

To run this project, you need to have Python and Jupyter Notebook installed. Clone the repository and install the necessary packages using:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

## Usage

Open the Jupyter Notebook to explore the code and experiment with the models. You can adjust model parameters and preprocessing techniques to see how they affect model performance.

```bash
jupyter notebook credit-card-fraud-prediction-rf-smote.ipynb
```

## Contributing

Contributions are welcome! If you have suggestions for improvement or additional features, please feel free to submit a pull request or open an issue.
