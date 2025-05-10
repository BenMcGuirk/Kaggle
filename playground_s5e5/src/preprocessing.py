from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

caps = {}
def clean_train_data(data):
    # Convert age to float
    data['Age'] = data['Age'].astype(float)

    # Binary encode sex
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    
    # Handle duplicate columns - aggregate target values
    features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    data = data.groupby(features, as_index=False)['Calories'].mean()

    # Cap outliers
    for col in ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']:
        lower = data[col].quantile(0.01)
        upper = data[col].quantile(0.99)
        caps[col] = (lower, upper)
        data[col] = data[col].clip(lower, upper)

    return data

def clean_test_data(data):
    # Convert age to float
    data['Age'] = data['Age'].astype(float)

    # Binary encode sex
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

    # Cap outliers
    for col in ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']:
        lower, upper = caps[col]
        data[col] = data[col].clip(lower, upper)

    return data

def build_preprocessor(X):
    # Identify column types
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Define which columns should use which scalers
    robust_cols = [
        'Age',           # Can have outliers, not normally distributed
        'Height',        # Can have outliers, not normally distributed
        'Weight',        # Can have outliers, not normally distributed
        'BMI',           # Derived from Height and Weight, inherits their properties
        'BMR',           # Derived from multiple features, can have outliers
        'Duration_per_Weight',  # Ratio, can have outliers
        'Duration_per_Height',  # Ratio, can have outliers
        'Duration_per_Age',     # Ratio, can have outliers
        'Weight_Duration'       # Interaction term, can have outliers
    ]
    
    minmax_cols = [
        'Duration',      # Has natural lower bound (0)
        'Heart_Rate',    # Has natural bounds (resting to max)
        'HR_Intensity',  # Already between 0 and 1
        'Max_HR',        # Has natural bounds
        'Fat_Burn_Zone', # Derived from BMR, has natural bounds
        'Fat_Burn_Zone_Min', # Has natural bounds
        'Fat_Burn_Zone_Max',  # Has natural bounds
        'Exercise_Intensity'  # Derived from HR_Intensity and Duration, has natural bounds
    ]
    
    standard_cols = [
        'Duration_HR'    # Interaction term, might be normally distributed
    ]

    used_num_cols = set(robust_cols + minmax_cols + standard_cols)
    fallback_cols = list(set(num_cols) - used_num_cols)
    
    # Build pipelines
    robust_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    minmax_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    standard_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    fallback_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('robust', robust_pipeline, robust_cols),
        ('minmax', minmax_pipeline, minmax_cols),
        ('standard', standard_pipeline, standard_cols),
        ('fallback', fallback_pipeline, fallback_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    return preprocessor

def fit_and_transform_preprocessor(preprocessor, X_train, X_val):
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    return X_train_processed, X_val_processed

def create_features(X):
    # BMI   
    X['BMI'] = X['Weight'] / (X['Height'] ** 2)

    # BMR
    X['BMR'] = np.where(
        X['Sex'] == 1,  # male
        88.362 + (13.397 * X['Weight']) + (4.799 * X['Height']) - (5.677 * X['Age']),
        447.593 + (9.247 * X['Weight']) + (3.098 * X['Height']) - (4.330 * X['Age'])
    )    

    # Exercise intensity
    X['Max_HR'] = 220 - X['Age']
    X['HR_Intensity'] = X['Heart_Rate'] / X['Max_HR']
    X['Exercise_Intensity'] = X['HR_Intensity'] * X['Duration'] * X['Body_Temp']

    # Fat burn zone
    X['Fat_Burn_Zone'] = X['BMR'] * 0.55
    X['Fat_Burn_Zone_Min'] = X['Fat_Burn_Zone'] * 0.8
    X['Fat_Burn_Zone_Max'] = X['Fat_Burn_Zone'] * 0.85

    # Duration per Weight
    X['Duration_per_Weight'] = X['Duration'] / X['Weight']
    # Duration per Height
    X['Duration_per_Height'] = X['Duration'] / X['Height']
    # Duration per Age
    X['Duration_per_Age'] = X['Duration'] / X['Age']

    # Age Groups
    X['Age_Group'] = pd.cut(
        X['Age'],
        bins=[0, 30, 45, 60, 100],
        labels=['18-30', '31-45', '46-60', '60+']
    )
    
    # Interaction Terms
    X['Duration_HR'] = X['Duration'] * X['Heart_Rate']
    X['Weight_Duration'] = X['Weight'] * X['Duration']

    # BMR * Duration
    X['BMR_Duration'] = X['BMR'] * X['Duration']
    X['BMR_Duration_Exercise_Intensity'] = X['BMR_Duration'] * X['Exercise_Intensity']

    # Duration * Heart Rate
    X['Duration_HR'] = X['Duration'] * X['Heart_Rate']
    X['Duration_HR_Exercise_Intensity'] = X['Duration_HR'] * X['Exercise_Intensity']

    return X