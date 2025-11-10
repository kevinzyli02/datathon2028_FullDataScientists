# prediction_example.py
"""
EXAMPLE OF USING THE TRAINED MODEL FOR PREDICTIONS
"""
import pandas as pd


def example_prediction():
    """Example of how to use the trained model"""

    # Load the saved model and artifacts
    model = joblib.load('optimized_xgboost_model/optimized_xgboost_model.pkl')
    imputer = joblib.load('optimized_xgboost_model/imputer.pkl')

    # Load feature names
    with open('optimized_xgboost_model/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]

    # Example new data (you would load your actual new data)
    new_data = pd.DataFrame({
        'stress_score': [75, 80, 65],
        'lightly': [120, 110, 130],
        'moderately': [45, 50, 40],
        'very': [30, 25, 35],
        'minutesawake': [15, 20, 10],
        'resting_hr': [65, 68, 62],
        'minutesasleep': [420, 400, 440]
        # Derived features will be automatically created
    })

    # Make predictions
    predictions = predict_hormones(model, imputer, new_data)

    print("🎯 Hormone Predictions:")
    print(predictions)

    return predictions

# Run the example
# predictions = example_prediction()