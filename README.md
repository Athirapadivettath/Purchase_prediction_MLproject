# Purchase Intention Prediction & Behavior Analysis

Author: Athira P

Project Overview:
This project builds a machine learning pipeline to predict whether an online visitor will make a purchase (Revenue = True) on an e-commerce website. The project also provides insights into visitor behavior and identifies key features influencing purchase decisions.

Project Aim

To develop a predictive model that forecasts customer purchase intention using session-level and visitor behavior data. The insights help e-commerce platforms optimize marketing strategies, website experience, and improve conversion rates.

Objectives

Analyze visitor session and behavioral data.

Detect patterns in numerical and categorical features influencing purchase behavior.

Build, train, and evaluate multiple machine learning models.

Handle skewness, outliers, and class imbalance for better model performance.

Generate actionable insights and predicted customer lists for business decisions.

Dataset

Source: Online Shoppers Intention Dataset
Size: 12,205 rows × 18 columns (after duplicate removal)

Key Features:

Numerical: Administrative, Informational, ProductRelated, BounceRates, ExitRates, PageValues, SpecialDay, OperatingSystems, Browser, Region, TrafficType

Categorical: Month, VisitorType, Weekend

Target: Revenue (True if purchase made, False otherwise)

Project Workflow
1. Install & Import Libraries

Libraries used: pandas, numpy, seaborn, matplotlib, scikit-learn, imblearn, lightgbm, xgboost.

2. Load & Preprocess Data

Handled missing values and duplicates.

Detected and corrected skewness using log/sqrt, Yeo-Johnson, and QuantileTransformer.

Handled outliers using Z-score method.

Separated numerical and categorical features.

Scaled numerical features and encoded categorical features.

3. Train-Test Split & Imbalance Handling

Split data (80/20) into train and test sets.

Balanced the training set using SMOTE to handle class imbalance.

4. Model Training & Evaluation

Models trained: Logistic Regression, Random Forest, LightGBM.

Evaluated using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.

LightGBM selected as best-performing model for final predictions.

5. Feature Importance & Insights

Extracted top predictive features using LightGBM.

Key features include ProductRelated_Duration, ExitRates, PageValues, and Administrative_Duration.

Visualized feature importance and generated actionable insights.

6. Final Predicted Customers

Predicted customers with high purchase intention.

Generated summary table, distribution plot, and exported predicted customers to CSV for reporting.

Metric	Value
Total Test Customers	2,441
Predicted to Purchase	383
Predicted Not to Purchase	2,058
Predicted Purchase %	15.7%

Plots generated: Predicted purchase distribution, top 15 features impacting purchase.

Usage

Clone the repository:

git clone <your-repo-url>


Open Purchase_Prediction.ipynb in Google Colab or Jupyter Notebook.

Upload Online purchase.csv.

Run all cells sequentially to preprocess data, train models, and generate predictions.

The predicted customer list is saved as predicted_customers.csv.

Key Insights

Majority of visitors do not purchase (84.3%).

Returning visitors and users spending more time on product-related pages are more likely to make a purchase.

LightGBM performed best overall with balanced precision and recall.

Predicted customers provide a target list for marketing campaigns.

Future Improvements

Include behavioral clustering for advanced segmentation.

Integrate real-time prediction using streaming visitor data.

Test additional ensemble or deep learning models for improved accuracy.

License
