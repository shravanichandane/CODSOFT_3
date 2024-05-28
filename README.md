# TASK - 3

## Evaluation of Customer Churn Prediction Code

#### Overview
A detailed analysis and prediction of customer churn using machine learning techniques. This project aims to provide actionable insights to businesses for retaining customers and minimizing revenue loss.

### Dataset
The dataset used for this analysis is [Churn_Modelling.csv](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction), which contains information about customers including their demographics, banking details, and churn status.

### Code File
The code for customer churn prediction is provided [HERE](https://github.com/shravanichandane/CODSOFT_3/blob/main/CODSOFT_3.ipynb) in the same respository.

### 1. Data Loading and Cleaning
- **Data Source**: The code successfully loads the dataset from a given file path.
- **Data Integrity Check**: It checks for missing values and duplicated rows, ensuring data integrity.
- **Result**: The dataset is clean and ready for analysis without any missing values or duplicates.

### 2. Data Preparation and Visualization
- **Categorical Encoding**: Categorical variables like 'Geography' and 'Gender' are encoded using LabelEncoder to convert them into numerical format, making them suitable for machine learning algorithms.
- **Visualizations**: The code provides visualizations such as histograms and boxplots to explore the distribution of numerical features and relationships between features and the target variable ('Exited').
- **Insight**: These visualizations offer insights into the distribution of features and potential patterns related to customer churn.

### 3. Model Training and Evaluation
- **Model Selection**: Logistic Regression, Random Forest, and Gradient Boosting classifiers are chosen for modeling customer churn.
- **Training and Evaluation**: Models are trained on the dataset and evaluated using cross-validation to ensure robustness. Evaluation metrics such as accuracy, precision, recall, and F1-score are computed to assess model performance.
- **Model Evaluation Results**:
  - **Logistic Regression**:
    - Accuracy: 0.8155
    - Precision: 0.6000
    - Recall: 0.1832
    - F1 Score: 0.2807
  - **Random Forest**:
    - Accuracy: 0.8640
    - Precision: 0.7469
    - Recall: 0.4656
    - F1 Score: 0.5737
  - **Gradient Boosting**:
    - Accuracy: 0.8660
    - Precision: 0.7551
    - Recall: 0.4707
    - F1 Score: 0.5799
- **Result**: The trained models provide baseline performance metrics for predicting customer churn, with Random Forest and Gradient Boosting outperforming Logistic Regression in terms of accuracy and F1-score.

### 4. Feature Importance Analysis
- **Analysis Method**: Feature importance is analyzed using a Random Forest classifier to identify key features contributing to customer churn prediction.
- **Result**: The importance of each feature is determined, providing insights into which features are most influential in predicting churn behavior.

### 5. Model Interpretation
- **SHAP Values**: SHAP (SHapley Additive exPlanations) values are computed to interpret model predictions and understand the impact of each feature on predicting customer churn.
- **Interpretation**: The SHAP values offer a deeper understanding of how individual features affect the model's predictions, aiding in model interpretation and decision-making.

### 6. Customer Segmentation
- **Clustering Analysis**: K-means clustering is applied to segment customers based on their attributes.
- **Cluster Analysis**: Characteristics of each customer cluster are analyzed to identify distinct segments and potential patterns related to churn behavior.
- **Insight**: Customer segmentation provides actionable insights for targeted retention strategies and personalized marketing approaches.

### Conclusion
The provided code implements a comprehensive analysis pipeline for customer churn prediction, including data loading, cleaning, modeling, evaluation, feature importance analysis, model interpretation, and customer segmentation. The results obtained from these analyses offer valuable insights for businesses to understand customer churn behavior and implement effective retention strategies. Random Forest and Gradient Boosting models demonstrate superior performance compared to Logistic Regression, with higher accuracy and F1-score, indicating their effectiveness in predicting customer churn.
