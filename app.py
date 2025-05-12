from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Load model, scaler, and selector
try:
    model = joblib.load('rf_model.pkl')
    selector = joblib.load('feature_selector.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model, selector, and scaler loaded successfully")
except Exception as e:
    print(f"Error loading files: {e}")

@app.route('/')
def home():
    print("Accessing root route")
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index.html: {e}")
        return jsonify({'error': 'Failed to render index.html'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        print("Received features:", features)

        # Validate input
        if not features or len(features) != 7:
            return jsonify({'error': 'Expected 7 features'}), 400
        try:
            features = [float(x) for x in features]
        except ValueError:
            return jsonify({'error': 'All features must be numeric'}), 400

        # Input features (7 selected features)
        input_feature_names = [
            'mean radius', 'mean perimeter', 'mean concave points',
            'worst radius', 'worst perimeter', 'worst area', 'worst concave points'
        ]
        features_array = np.array(features).reshape(1, -1)
        features_df = pd.DataFrame(features_array, columns=input_feature_names)
        print("Input Features DataFrame:", features_df.to_dict())

        # Create full feature set (30 features)
        all_feature_names = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
            'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
            'area error', 'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
            'worst fractal dimension'
        ]
        full_features_df = pd.DataFrame(np.zeros((1, 30)), columns=all_feature_names)
        for col in input_feature_names:
            full_features_df[col] = features_df[col]
        print("Full Features DataFrame:", full_features_df.to_dict())

        # Scale features
        features_scaled = scaler.transform(full_features_df)
        print("Scaled features:", features_scaled)

        # Apply feature selector
        features_selected = selector.transform(features_scaled)
        print("Selected features:", features_selected)

        # Make prediction
        prediction = model.predict(features_selected)[0]
        print("Prediction:", prediction)

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        print("Error in /predict:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)