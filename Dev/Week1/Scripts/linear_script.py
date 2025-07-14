import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def fill_missing_rm(data):
    """Fill missing values in 'rm' column with the mean."""
    data['rm'] = data['rm'].fillna(data['rm'].mean())
    return data

def remove_outliers(df, factor=2.5):
    """Remove outliers using the IQR method for all numeric columns."""
    df_clean = df.copy()
    for column in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Split data and scale features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_linear_regression(X_train_scaled, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model

def save_artifacts(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    """Save the trained model and scaler."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_artifacts(model_path='model.pkl', scaler_path='scaler.pkl'):
    """Load the trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler