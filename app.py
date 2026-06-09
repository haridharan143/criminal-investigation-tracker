from flask import Flask, render_template, request, redirect # type: ignore
import sqlite3
import pickle
import numpy as np
import os

app = Flask(__name__)
DB = 'database.db'

# Load ML model
model = pickle.load(open("model/predict_model.pkl", "rb"))

# Setup database
def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_name TEXT,
            suspect_name TEXT,
            evidence TEXT
        )''')
init_db()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/add", methods=["GET", "POST"])
def add_case():
    if request.method == "POST":
        case_name = request.form['case_name']
        suspect_name = request.form['suspect_name']
        evidence = request.form['evidence']
        with sqlite3.connect(DB) as conn:
            conn.execute("INSERT INTO cases (case_name, suspect_name, evidence) VALUES (?, ?, ?)",
                         (case_name, suspect_name, evidence))
        return redirect("/cases")
    return render_template("add_case.html")


@app.route("/cases")
def view_cases():
    with sqlite3.connect(DB) as conn:
        cursor = conn.execute("SELECT * FROM cases")
        cases = cursor.fetchall()
    return render_template("view_cases.html", cases=cases)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        prior = int(request.form['prior'])
        gang = int(request.form['gang'])
        violent = int(request.form['violent'])

        features = np.array([[prior, gang, violent]])
        prediction = model.predict(features)
        result = "LIKELY SUSPECT" if prediction[0] == 1 else "UNLIKELY"
    return render_template("prediction.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
