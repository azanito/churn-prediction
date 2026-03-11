# Bank Customer Churn Prediction

This project analyzes and predicts customer churn using machine learning techniques.

## Dataset
Bank Customer Churn dataset from Kaggle:  
https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

The dataset contains information about bank clients such as credit score, geography, age, balance, and activity status.  
The goal is to predict whether a customer will leave the bank (`Exited` column).

## Project Steps

1. **Exploratory Data Analysis (EDA)**
   - Target variable distribution
   - Correlation matrix of numerical features

2. **Data Preprocessing**
   - Removal of identifier columns
   - Encoding categorical features using `LabelEncoder`
   - Train/test split

3. **Model Training**
   - Baseline model: Random Forest
   - Advanced model: XGBoost with `GridSearchCV`

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - MAE

5. **Feature Importance**
   - XGBoost feature importance plots

6. **Model Explainability**
   - SHAP summary plot
   - SHAP feature importance
   - SHAP waterfall explanations for individual predictions

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- SHAP
- matplotlib
- seaborn

## Outputs

All generated plots are saved in the `outputs/` directory.
