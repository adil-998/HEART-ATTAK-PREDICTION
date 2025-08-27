❤️ Heart Attack Prediction

This project aims to predict the likelihood of a heart attack based on various medical indicators using machine learning techniques. The model is built and tested in a Jupyter Notebook with Python and popular data science libraries such as pandas, numpy, scikit-learn, matplotlib, and seaborn.

📌 Project Overview

The goal of this project is to develop a binary classification model that predicts whether a patient is at risk of a heart attack (positive) or not (negative).

Key Steps

Data Loading & Exploration

Load dataset, check shape, column types, and preview rows.

Data Preprocessing

Handle missing values and duplicates.

Convert categorical labels (Result) into numeric format (0 = negative, 1 = positive).

Convert Blood sugar to integer type.

Outlier Detection & Removal

Use the IQR method to detect and remove outliers in Age, Heart rate, CK-MB, and Troponin.

Exploratory Data Analysis (EDA)

Histogram of Blood sugar.

Bar plot of Gender vs. Result.

Bar plot of Top 5 age groups with positive results.

Scatter plot of CK-MB vs. Troponin.

Correlation heatmap for feature relationships.

Model Building

Split dataset into training and testing sets.

Train models:

Logistic Regression (baseline, ~65% accuracy).

Random Forest Classifier (best performance, ~97% accuracy).

Model Evaluation

Accuracy, Precision, Recall, F1 Score

Confusion Matrix for detailed performance evaluation.

📊 Key Findings

Logistic Regression gave a reasonable baseline (~65% accuracy) but showed convergence warnings → feature scaling helped but performance remained limited.

Random Forest Classifier achieved 97% accuracy, making it the best-suited model for this dataset.

EDA revealed strong relationships between certain health metrics and the likelihood of a heart attack.

⚙️ Installation & Setup

Make sure you have Python 3.7+ installed. Install the required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn jupyter

🚀 How to Run

Clone this repository or download the files.

Open the terminal and launch Jupyter Notebook:

jupyter notebook


Open heart_attack_prediction.ipynb in the Jupyter interface.

Run all cells to see the data analysis, visualizations, and model results.

🛠️ Tech Stack

Python

pandas & numpy – data processing

matplotlib & seaborn – visualization

scikit-learn – machine learning models & evaluation

📌 Future Improvements

Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV).

Test other ML algorithms like XGBoost, SVM, Neural Networks.

Build a web app (Streamlit/Flask) for user-friendly prediction.

