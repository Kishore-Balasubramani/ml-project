# train.py

import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from utils import load_data

def train_model():
    # Load dataset
    X, y = load_data()

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X, y)

    # Save trained model
    joblib.dump(model, "model.pkl")
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    train_model()