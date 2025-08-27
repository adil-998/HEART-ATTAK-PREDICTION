Heart Attack Prediction
This project aims to predict the likelihood of a heart attack based on various medical indicators. The analysis and model development are performed using a Jupyter Notebook with Python and popular data science libraries such as pandas, numpy, scikit-learn, matplotlib, and seaborn.

Project Overview
The core of this project is a machine learning model that uses a dataset containing patient information to predict a binary outcome: whether a heart attack is likely ("positive") or not ("negative").

The key steps in this project include:

Data Loading and Initial Exploration: Reading the dataset and getting a first look at the data, including its shape, column types, and a few sample rows.

Data Preprocessing: Cleaning and preparing the data for model training. This includes checking for missing values, handling duplicates, and converting the 'Result' column from categorical text to a numerical format (0 for 'negative' and 1 for 'positive'). The Blood sugar column was also converted to an integer type.

Outlier Detection and Removal: Using the Interquartile Range (IQR) method to identify and remove outliers in columns like 'Age', 'Heart rate', 'CK-MB', and 'Troponin'. This step is crucial for improving model accuracy by ensuring the data is within a realistic range.

Exploratory Data Analysis (EDA): Visualizing relationships between key variables. This includes:

A histogram of 'Blood sugar' to understand its distribution.

A bar plot showing the relationship between 'Gender' and the 'Result'.

A bar plot highlighting the top 5 age groups associated with positive results.

A scatter plot visualizing the relationship between 'CK-MB' and 'Troponin' levels.

A correlation heatmap to understand the relationships between different numerical features.

Model Building:

The dataset is split into training and testing sets.

A Logistic Regression model is trained to establish a baseline.

A Random Forest Classifier is then trained, which significantly improves prediction accuracy.

Model Evaluation: The models are evaluated using several metrics to assess their performance:

Accuracy: The proportion of correctly predicted instances.

Precision: The proportion of true positive results among all positive predictions.

Recall: The proportion of true positive results that were correctly identified.

F1 Score: The harmonic mean of precision and recall.

Confusion Matrix: A table showing the number of true positives, true negatives, false positives, and false negatives.

Key Findings
The initial exploratory analysis reveals important relationships between different health metrics and the prediction result.

The Logistic Regression model provides a reasonable baseline but shows a warning about convergence, which suggests that the features may need scaling. Even after scaling, the accuracy of the Logistic Regression model was about 65%.

The Random Forest Classifier model demonstrates superior performance with an accuracy of over 97%, indicating it is well-suited for this prediction task.

How to Run the Notebook
To run this project, you will need a Python environment with the following libraries installed:

numpy

pandas

matplotlib

seaborn

scikit-learn

jupyter

You can install these libraries using pip:

Bash

pip install numpy pandas matplotlib seaborn scikit-learn jupyter
Once the dependencies are installed, you can open and run the heart attack prediction.ipynb notebook using Jupyter:

Bash

jupyter notebook
This will open a web browser where you can navigate to the notebook file and execute the cells in sequence.
