# RecoverIQ – Debt Recovery Prediction System

RecoverIQ is a machine learning-based debt recovery prediction system designed to predict whether a customer is likely to repay an outstanding debt. It uses historical customer data and machine learning techniques to support data-driven debt recovery decisions.

## Overview

RecoverIQ applies supervised machine learning to analyze customer information and estimate the likelihood of debt repayment. The system uses Logistic Regression as its selected prediction model.

## Key Features

- Customer debt repayment prediction
- Machine learning-based classification
- Data preprocessing and feature analysis
- Model training and evaluation
- Probability-based prediction using a configurable classification threshold
- Model performance evaluation using multiple classification metrics

## Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Visualization:** Matplotlib
- **Application Interface:** Streamlit

## Machine Learning Approach

RecoverIQ follows a supervised learning approach.

**Model Selection:**

Two classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest Classifier

Logistic Regression was selected based on the comparative evaluation.

**Dataset Splitting:**

The dataset was divided into three subsets:

- Training Set: 60%
- Validation Set: 20%
- Testing Set: 20%

**Classification Threshold:**

A threshold of 0.6 was selected for the final prediction evaluation.

## Model Evaluation

The selected Logistic Regression model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 78.10% |
| Precision | 34.94% |
| Recall | 60.42% |
| F1-Score | 44.27% |
| ROC-AUC | 78.37% |
| Average Precision | 39.62% |

These metrics help evaluate the model's ability to identify customers who are likely to repay their outstanding debts.

## Project Structure

```text
RecoverIQ/
│
├── data/
│
├── models/
│   └── recovery_model.joblib
│
├── reports/
│
├── src/
│   └── evaluate_model.py
│
├── app.py
└── README.md
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/RecoverIQ.git
```

**2. Navigate to the project directory**

```bash
cd RecoverIQ
```

**3. Install dependencies**

```bash
pip install pandas numpy scikit-learn matplotlib joblib streamlit
```

**4. Run the application**

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Future Enhancements

- Experiment with additional machine learning algorithms.
- Improve prediction precision and model performance.
- Incorporate additional customer-related features.
- Enhance the user interface and data visualization.
- Explore advanced machine learning techniques for debt recovery prediction.

## Author

**Suha Maryam**

GitHub: [Your GitHub Profile](https://github.com/suha0)

---

*RecoverIQ – Making debt recovery decisions more data-driven through machine learning.*
