#!/usr/bin/env python
# coding: utf-8

# # What drives the price of a car? Analysis
# 
# Applying the CRISP-DM framework to understand the key drivers of used car prices.

# ## 1. Load and Explore Data

# In[8]:


get_ipython().system('pip install missingno')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import missingno as msno  # Library for missing values visualization

# Load the dataset

file_path = "/Users/anumita/Downloads/practical_application_II_starter/data/vehicles.csv"
df = pd.read_csv(file_path)
df.head()

# Display dataset structure
print("Dataset Overview:")
print(df.info())

# Check missing values
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# Visualize missing values
msno.bar(df)
plt.title("Missing Values in Dataset")
plt.show()

# Display first five rows
df.head()


# ## 2. Data Cleaning & Preparation

# In[9]:


# Drop unnecessary columns
df_cleaned = df.drop(columns=['id', 'VIN'])
# Handle missing values
df_cleaned = df_cleaned.dropna(subset=['price', 'year'])
categorical_columns = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color']
for col in categorical_columns:
    df_cleaned[col] = df_cleaned[col].fillna('Unknown')

df_cleaned['odometer'].fillna(df_cleaned['odometer'].median(),inplace=True)
df_cleaned['year'] = df_cleaned['year'].astype(int)
df_cleaned['car_age'] = 2025 - df_cleaned['year']
df_cleaned.info()


# ## 3. Exploratory Data Analysis (EDA)

# In[10]:


plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['price'], bins=100, kde=True)
plt.xlim(0, df_cleaned['price'].quantile(0.99))
plt.title("Distribution of Car Prices")
plt.xlabel("Price ($)")
plt.ylabel("Frequency")
plt.show()


# ## 4. Data Preprocessing

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

file_path = "/Users/anumita/Downloads/practical_application_II_starter/data/vehicles.csv"
df= pd.read_csv(file_path)
df.head()

# Data Cleaning
df_cleaned = df.copy() #copy for avoiding modifying dataset

# Drop unnecessary columns
df_cleaned =df_cleaned.drop(columns=['id','VIN'], errors='ignore')

# Drop rows with missing $ or year
df_cleaned=df_cleaned.dropna(subset=['price','year'])

# Missing value with "Unknown"
categorical_columns = ['manufacturer','fuel', 'transmission','drive','type','paint_color']
for col in categorical_columns:
    df_cleaned[col]= df_cleaned[col].fillna('Unknown')

# Missing od values with median
df_cleaned['odometer'].fillna(df_cleaned['odometer'].median(),inplace=True)

#'Year' column to integer
df_cleaned['year'] =df_cleaned['year'].astype(int)

# New column for car age
df_cleaned['car_age']=2025 - df_cleaned['year']  # Assuming the current year is 2025

# Verify
df_cleaned.info()
df_cleaned.head()


# ## 5. Model Training & Evaluation

# In[ ]:


# Import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Feature set (X) and target variable (y)
X = df_cleaned[['car_age', 'odometer', 'manufacturer', 'fuel', 'transmission', 'drive', 'type', 'paint_color']]
y =df_cleaned['price']

# Train-test split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical and numerical features
categorical_features=['manufacturer', 'fuel', 'transmission', 'drive', 'type', 'paint_color']
numerical_features = ['car_age','odometer']

# Preprocessing
preprocessor =ColumnTransformer(transformers=[
    ('num',StandardScaler(), numerical_features), 
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False),categorical_features)  
])

# Fit and transform
X_train_preprocessed= preprocessor.fit_transform(X_train)
X_test_preprocessed=preprocessor.transform(X_test)

# Confirm after preprocessing
print("Training Data Shape:",X_train_preprocessed.shape)
print("Testing Data Shape:", X_test_preprocessed.shape)

# Initialize
lr = LinearRegression()
rf=RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train
lr.fit(X_train_preprocessed, y_train)
rf.fit(X_train_preprocessed,y_train)
gbr.fit(X_train_preprocessed, y_train)

# Make predictions
y_pred_lr= lr.predict(X_test_preprocessed)
y_pred_rf =rf.predict(X_test_preprocessed)
y_pred_gbr = gbr.predict(X_test_preprocessed)

# Evaluate
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse =np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return pd.DataFrame([[model_name,mae, rmse, r2]], columns=['Model','MAE','RMSE', 'R² Score'])

# Collect
results_df = pd.concat([
    evaluate_model(y_test,y_pred_lr, "Linear Regression"),
    evaluate_model(y_test, y_pred_rf, "Random Forest"),
    evaluate_model(y_test,y_pred_gbr, "Gradient Boosting")
], ignore_index=True)

# Display
import ace_tools as tools
tools.display_dataframe_to_user(name="Model Evaluation Results", dataframe=results_df)


# 

# In[ ]:





# In[ ]:




