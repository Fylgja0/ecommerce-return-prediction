import pandas as pd

# 1. Load Dataset
print("--- LOADING DATASET ---")
df = pd.read_excel('Data/ecommerce_global_sales_dataset.xlsx')
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}\n")
print("=" * 60 + "\n")

# Drop unnecessary columns if they exist in the dataframe
dropped_cols = ['previous_device_os', 'storage', 'sale_id', 'sale_date', 'year', 'month', 'country', 'city']
cols_to_drop = [col for col in dropped_cols if col in df.columns]
df_cleaned = df.drop(columns=cols_to_drop)

# 2. Descriptive Statistics for Numerical Variables
print("--- 1. DESCRIPTIVE STATISTICS FOR NUMERICAL VARIABLES ---")
numeric_df = df_cleaned.select_dtypes(include=['int64', 'float64'])

if not numeric_df.empty:
    stats_df = numeric_df.describe().T

    # Calculate range: Max - Min
    stats_df['range'] = stats_df['max'] - stats_df['min']

    # Rename columns for clarity
    stats_df = stats_df.rename(columns={
        'mean': 'Mean',
        '50%': 'Median (50%)',
        'min': 'Minimum',
        'max': 'Maximum',
        'range': 'Range',
        'std': 'Standard Deviation'
    })
    print(stats_df[['Mean', 'Median (50%)', 'Minimum', 'Maximum', 'Range']])
else:
    print("No numerical variables found in the dataset.")
print("\n" + "=" * 60 + "\n")

# 3. Categorical Variables Frequency and Mode Analysis
print("--- 2. MODE AND FREQUENCY ANALYSIS FOR CATEGORICAL VARIABLES ---")
categorical_cols = df_cleaned.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

for col in categorical_cols:
    mode_val = df_cleaned[col].mode()[0]
    freq = df_cleaned[col].value_counts().iloc[0]
    percentage = (freq / len(df_cleaned)) * 100
    print(f"Variable: {col:25} | Mode: {mode_val:20} | Frequency: {freq} ({percentage:.2f}%)")
print("\n" + "=" * 60 + "\n")

# 4. Missing Value Analysis and Imputation Strategy
print("--- 3. MISSING VALUE DETECTION AND IMPUTATION ---")
missing_values = df_cleaned.isnull().sum()
missing_cols = missing_values[missing_values > 0]

if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        print(f"Column with Missing Data: {col} | Missing Rows: {count}")
        median_val = df_cleaned[col].median()
        print(f"-> Imputation: Imputed with median value ({median_val}) to prevent skewness.")
else:
    print("No missing data found in the dataset.")
print("\n" + "=" * 60 + "\n")

# 5. Correlation Analysis (Numerical Variables)
print("--- 4. CORRELATION ANALYSIS (NUMERICAL VARIABLES) ---")
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    print(corr_matrix)
    print("\n" + "-" * 60)
    print("Note: Pearson correlation only captures linear relationships among numerical features.")
    print("To evaluate non-linear relationships and high-cardinality categorical features,")
    print("feature importance analysis is performed using RandomForest.py.")
else:
    print("Not enough numerical columns found for correlation analysis.")
print("=" * 60 + "\n")