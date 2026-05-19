import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from catboost import CatBoostClassifier

# 1. Load and Clean Data
df = pd.read_excel('Data/ecommerce_global_sales_dataset.xlsx')
df = df.drop(columns=['previous_device_os', 'storage', 'sale_id', 'sale_date', 'year', 'month', 'country', 'city'])

df['customer_rating'] = df['customer_rating'].fillna(df['customer_rating'].median())

# 2. Separate features (X) and target (y)
X = df.drop('return_status', axis=1)
y = df['return_status']

# 2. Automatically determine categorical columns.
categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
print(f"Number of Categorical Features Detected Automatically: {len(categorical_features)}")
print(f"Identified Categorical Features: {categorical_features}\n")
print("-" * 50)

# 3. Train-Test Split (80% Training, 20% Testing)
# Stratify ensures proportional representation of imbalanced classes in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train CatBoost Classifier with GPU Acceleration
print("\nTraining CatBoost Classifier...")

cb_model = CatBoostClassifier(
    iterations=1000,                     # Count of boosting iterations
    learning_rate=0.03,                  # Learning rate
    depth=6,                             # Tree depth
    cat_features=categorical_features,   # Inform the algorithm about categorical columns
    auto_class_weights='Balanced',       # Automatically handles imbalanced data
    loss_function='MultiClass',          # Suitable for multi-class classification
    task_type='GPU',                 
    random_seed=42,
    verbose=100                          # Print training progress every 100 iterations
)

# Fit the model
cb_model.fit(X_train, y_train)

# 5. Estimation and Evaluation
y_pred = cb_model.predict(X_test)

print("\n" + "="*50)
print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Scores
importance = cb_model.get_feature_importance()
for i, feature in enumerate(X.columns):
    print(f"{feature}: {importance[i]:.2f}%")