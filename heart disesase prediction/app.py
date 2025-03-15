from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("heart.csv") 


X = df[['age', 'sex', 'cp', 'trestbps', 'chol']]
y = df['target']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received Data:", data)

        age = float(data["age"])
        sex = int(data["sex"])
        cp = int(data["cp"])
        trestbps = float(data["trestbps"])
        chol = float(data["chol"])

     
        input_data = np.array([[age, sex, cp, trestbps, chol]])
        input_scaled = scaler.transform(input_data)

       
        prediction = model.predict(input_scaled)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        
        print("Prediction:", result)
        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
