from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
with open('RandomForestClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()
        features = data['features']  # Expecting {"features": [values]}
        
        # Make prediction
        prediction = model.predict([features])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
