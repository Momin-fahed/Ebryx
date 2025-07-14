import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

def load_iris_data(filepath):
    """Load the Iris dataset from a CSV file and drop the 'Id' column if present."""
    df = pd.read_csv(filepath)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    return df

def check_missing(df):
    """Print missing value counts for each column."""
    print(df.isnull().sum())

def preprocess_features(df):
    """Split features and target."""
    X = df.drop('Species', axis=1)
    y = df['Species']
    return X, y

def encode_target(y):
    """Encode target labels."""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Split data and scale features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic_regression(X_train, y_train, max_iter=200):
    """Train a logistic regression model."""
    model = LogisticRegression(solver='lbfgs', max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Print classification report."""
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def save_artifacts(model, scaler, encoder, prefix='iris'):
    """Save model, scaler, and encoder."""
    joblib.dump(model, f'{prefix}_model.pkl')
    joblib.dump(scaler, f'{prefix}_scaler.pkl')
    joblib.dump(encoder, f'{prefix}_encoder.pkl')

def load_artifacts(prefix='iris'):
    """Load model, scaler, and encoder."""
    model = joblib.load(f'{prefix}_model.pkl')
    scaler = joblib.load(f'{prefix}_scaler.pkl')
    encoder = joblib.load(f'{prefix}_encoder.pkl')
    return model, scaler, encoder

def reduce_dimensionality(X_train, X_test, n_components=2):
    """Reduce features to n_components using PCA."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca,