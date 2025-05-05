from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

caps = {}
def clean_data(data):
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
    
    # Build pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor

def fit_and_transform_preprocessor(preprocessor, X_train, X_val):
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    return X_train_processed, X_val_processed