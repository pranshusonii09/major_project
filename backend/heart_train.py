import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Create models directory if not exists
os.makedirs('./models', exist_ok=True)

# Function to train and save a model
def train_and_save_model(data_path, target_column, model_path):
    # Load the dataset
    data = pd.read_csv(data_path)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained for {os.path.basename(data_path)}. Accuracy: {accuracy:.2f}")

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

# Train Diabetes Prediction Model
train_and_save_model(
    data_path='.backend/diabetes.csv',
    target_column='Outcome',  # Change this if your dataset uses a different target name
    model_path='./models/diabetes_prediction_model.joblib'
)

# Train Heart Disease Prediction Model
train_and_save_model(
    data_path='./datasets/heart_disease.csv',
    target_column='target',  # Change this if your dataset uses a different target name
    model_path='./models/heart_disease_prediction_model.joblib'
)
