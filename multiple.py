from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

data = pd.read_csv("multi.csv")
X = data[['Hours', 'Sleep', 'Attendance']]
Y = data['Marks']



model = LinearRegression()
model.fit(X, Y)


@app.route('/')
def home():
    return render_template("multi.html")


# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    hours = float(request.form['hours'])
    sleep = float(request.form['sleep'])
    attendance = float(request.form['attendance'])

    new_data = pd.DataFrame(
        [[hours, sleep, attendance]],
        columns=['Hours', 'Sleep', 'Attendance']
    )

    prediction = model.predict(new_data)

    marks = round(prediction[0], 2)

    if marks >= 90:
        result = "Excellent Performance 🌟"

    elif marks >= 75:
        result = "Very Good Performance 👍"

    elif marks >= 50:
        result = "Good Performance 🙂"

    else:
        result = "Need Improvement 📚"

    return render_template(
        "multi.html",
        prediction=marks,
        result=result
    )

if __name__ == '__main__':
    app.run(debug=True)