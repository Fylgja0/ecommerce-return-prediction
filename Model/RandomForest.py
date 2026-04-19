import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# 1. Load and Clean Data
df = pd.read_excel('Data/ecommerce_global_sales_dataset.xlsx')
df = df.drop(columns=['previous_device_os', 'storage', 'sale_id', 'sale_date', 'year', 'month', 'country', 'city'])

# Fill missing values for numerical column with median to avoid outlier skewness
df['customer_rating'] = df['customer_rating'].fillna(df['customer_rating'].median())

# 2. Encode Categorical Variables
le = LabelEncoder()
df['return_status'] = le.fit_transform(df['return_status']) # 0: Exchanged, 1: Kept, 2: Returned

# Separate features (X) and target (y)
X = df.drop('return_status', axis=1)
y = df['return_status']

# Apply One-Hot Encoding to categorical features (Drop first to avoid dummy trap)
X = pd.get_dummies(X, drop_first=True)

# 3. Train-Test Split (80% Training, 20% Testing)
# Stratify ensures proportional representation of imbalanced classes in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =====================================================================
# PHASE 1: FEATURE IMPORTANCE ANALYSIS (BASE MODEL)
# Training a base model to identify signal-to-noise ratio in the dataset
# =====================================================================
base_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
base_model.fit(X_train, y_train)

# Extract and sort feature importances
importances = base_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance_Score': importances})
importance_df = importance_df.sort_values(by='Importance_Score', ascending=False)

print("--- FEATURE IMPORTANCE (SIGNAL-TO-NOISE RATIO) ---")
print(importance_df)
print("\n" + "="*50 + "\n")

# =====================================================================
# PHASE 2: HANDLING CLASS IMBALANCE WITH SMOTE
# =====================================================================
print(f"Training Set Size (Before SMOTE): {X_train.shape[0]}")

# Apply SMOTE only to the training data to prevent data leakage into the test set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training Set Size (After SMOTE): {X_train_smote.shape[0]}")
print("-" * 50)

# =====================================================================
# PHASE 3: TRAINING WITH SYNTHETICALLY BALANCED DATA
# =====================================================================
balanced_model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
balanced_model.fit(X_train_smote, y_train_smote)

# Predictions and Evaluation
y_pred_balanced = balanced_model.predict(X_test)

print(f"\nRandom Forest Accuracy (After SMOTE): {accuracy_score(y_test, y_pred_balanced):.4f}\n")
print("Classification Report (SMOTE Balanced):")
print(classification_report(y_test, y_pred_balanced, target_names=le.classes_))

# =====================================================================
# PHASE 4: DIMENSIONALITY REDUCTION & OPTIMIZED MODEL
# =====================================================================
print("\n" + "="*50)
print("STARTING RETRAINING WITH TOP CONCENTRATED FEATURES...")

# Select only high-quality features (Importance Score > 1%)
# This filters out the extreme noise caused by hundreds of one-hot encoded product names/colors
top_features = importance_df[importance_df['Importance_Score'] > 0.01]['Feature'].tolist()

print(f"Selected {len(top_features)} high-quality features out of {len(X_train.columns)} total columns.")
print(f"Active Features: {top_features}\n")

# Restrict the training and testing sets to only these top features
X_train_optimized = X_train_smote[top_features]
X_test_optimized = X_test[top_features]

# Train the final concentrated model to break the accuracy paradox
optimized_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
optimized_model.fit(X_train_optimized, y_train_smote)

# Final Predictions and Evaluation
y_pred_optimized = optimized_model.predict(X_test_optimized)

print(f"Optimized Model Accuracy (Noise Reduced): {accuracy_score(y_test, y_pred_optimized):.4f}\n")
print("Classification Report (Optimized Model):")
print(classification_report(y_test, y_pred_optimized, target_names=le.classes_))
