import pandas as pd

df = pd.read_excel('Data/ecommerce_global_sales_dataset.xlsx')

# Only retrieve columns that contain at least one missing data entry
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
