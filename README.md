# 🛒 E-Commerce Global Sales Analytics: Predicting Customer Return Behavior

![Python](https://img.shields.io/badge/Python-3.13.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-green?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Big Data Analytics - End-of-Year Project** > *An in-depth analysis of imbalanced e-commerce datasets, tackling the "Accuracy Paradox," and extracting actionable business insights.*

## 📌 Project Overview
In the e-commerce industry, customer returns significantly impact profitability and logistics. This project aims to build a machine learning pipeline to predict customer post-purchase behavior: **Kept, Returned, or Exchanged**. 

Rather than just training models to achieve high accuracy, this project focuses on a real-world data science challenge: **Imbalanced Data and the Accuracy Paradox**. It documents the complete journey from initial naive predictions to advanced dimensionality reduction and synthetic data generation.

## 📊 The Dataset & The Challenge
The dataset contains global e-commerce sales records, including features like `unit_price_usd`, `customer_rating`, product details, and demographics. 

**The Core Problem (Class Imbalance):**
* **Kept:** ~85.9% (Majority Class)
* **Returned:** ~9.2% (Minority Class)
* **Exchanged:** ~4.9% (Minority Class)

## 🚀 Methodology & The Data Science Journey

### Phase 1: The Accuracy Paradox
Initial models (Decision Tree, Logistic Regression, Random Forest) achieved an impressive **85.9% Accuracy**. However, a deep dive into the Confusion Matrix revealed the "Accuracy Paradox". The models learned nothing; they simply predicted "Kept" for every single customer, completely failing to identify returns and exchanges (Recall: 0.00).

### Phase 2: Tackling Imbalance & The "Curse of Dimensionality"
To force the model to recognize minority classes, I applied:
1. **Class Weighting:** Adjusted algorithms to penalize misclassifications of minority classes.
2. **SMOTE (Synthetic Minority Over-sampling Technique):** Mathematically generated synthetic data for the "Returned" and "Exchanged" classes to balance the training set (from 12,400 to 31,956 samples).

Despite balancing, the models struggled with *False Positives*. The reason? **One-Hot Encoding** had expanded the dataset to 198 columns (mostly sparse product names and colors), causing the model to drown in noise.

### Phase 3: Dimensionality Reduction (Feature Importance)
Using Random Forest's `feature_importances_`, I analyzed the signal-to-noise ratio of the dataset. I discarded 183 noisy columns and isolated the **Top 15 Concentrated Features** (e.g., `unit_price`, `revenue`, `customer_rating`, `age_group`). 

Training the model exclusively on these high-quality features successfully broke the paradox. The model finally started attempting to predict true returns and exchanges.

### Phase 4: Advanced Boosting with CatBoost (The Paradigm Shift)
To optimize the pipeline further and completely bypass the manual encoding/dimensionality reduction dilemma, I integrated a **CatBoost Classifier**. 
* **Native Categorical Handling:** CatBoost processes high-cardinality categorical variables directly without creating sparse matrices or exploding the feature space.
* **Automated Class Balancing:** Utilized `auto_class_weights='Balanced'` to handle the severe data skew dynamically during training.
* **GPU Acceleration:** Leveraged GPU task types to significantly accelerate training iterations on the entire feature set while using early stopping to eliminate overfitting.

## 📈 Key Results
* **Naive Model Accuracy:** 86% *(Fake performance; 0% Minority Recall)*
* **Optimized Concentrated Model Accuracy:** 78% *(Real performance; Successfully predicting minority behaviors)*
* **CatBoost Performance:** Outperformed traditional sparse-matrix pipelines by processing raw high-cardinality data directly with optimized multi-class loss tracking.

## 💡 Ultimate Business Insights (The "So What?")
The most valuable outcome of this analytical process is not just a predictive model, but a critical business insight. 

Despite applying state-of-the-art optimization techniques (SMOTE, Feature Importance, Native CatBoost Balancing), the model's ability to confidently isolate returns remains limited by the nature of the data.

**Mathematical Conclusion:** *The current features collected by the company (Customer Age, Item Price, Rating) do not contain a strong enough signal to explain return behavior on their own.*

**Recommendations for the Business:**
To build a highly accurate predictive engine, the company must begin tracking new data points such as:
* Delivery times (Did the package arrive late?)
* Customer support ticket history
* Packaging damage reports at delivery
* Exact reason for return (defective, compatibility issues, damaged, changed mind)

## 🛠️ Technologies Used
* **Language:** Python
* **Data Manipulation:** Pandas
* **Gradient Boosting:** CatBoost
* **Machine Learning:** Scikit-Learn (Random Forest, Logistic Regression, Decision Tree)
* **Imbalanced Data Handling:** Imbalanced-learn (SMOTE)
* **Business Intelligence & Data Visualization:** Microsoft Excel (Interactive Dashboard, Pivot Tables & Dynamic Slicers)

## ⚙️ How to Run the Project
*(Note: Only the final, optimized Random Forest and advanced Catboost models are provided in this repository to ensure a clean execution flow.)*

1. Clone the repository:
   ```bash
   git clone https://github.com/Fylgja0/ecommerce-return-prediction.git
   ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the optimized Random Forest script:
    ```bash
    python Model/RandomForest.py
    ```

4. Run the advanced Catboost script:
    ```
    python Model/Catboost.py
    ```

5. Explore the Interactive Business Dashboard:
    Open `ecommerce_sales_dashboard.xlsx` in Microsoft Excel to explore the interactive management panel. Use the slicers on the left panel (`Region`, `Payment Method`, `Sales Channel`, `Customer Age Group`) to filter the entire sales and return metrics dynamically.

## 👨‍💻 Author
**Ahmetcan Bağlı**
* **LinkedIn:** [linkedin.com/in/ahmetcanbagli](https://linkedin.com/in/ahmetcanbagli)
* **GitHub:** [@Fylgja0](https://github.com/Fylgja0)

## 📄 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute this project for educational and professional purposes.