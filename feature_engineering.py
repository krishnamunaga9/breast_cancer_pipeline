import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def feature_engineering(X_train, X_test):
    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    return X_train_poly, X_test_poly

if __name__ == '__main__':
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    
    X_train_poly, X_test_poly = feature_engineering(X_train, X_test)
    
    # Save engineered features if needed
    pd.DataFrame(X_train_poly).to_csv('X_train_poly.csv', index=False)
    pd.DataFrame(X_test_poly).to_csv('X_test_poly.csv', index=False)
