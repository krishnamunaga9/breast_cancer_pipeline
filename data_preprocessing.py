import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Load dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data()
    # Save processed data if needed
    pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
    pd.Series(y_train).to_csv('y_train.csv', index=False)
    pd.Series(y_test).to_csv('y_test.csv', index=False)
