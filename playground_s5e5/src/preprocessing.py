import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocess_data(data):
    # Separate features and target
    X = data.drop(columns=['id', 'Calories'])
    y = data['Calories']
    
    # Identify column types
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor

if __name__ == "__main__":
    data = pd.read_csv('train.csv')
    X_processed, y, preprocessor = preprocess_data(data)
    
    # Save preprocessor for later use
    import joblib
    joblib.dump(preprocessor, 'preprocessor.pkl')