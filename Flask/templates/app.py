from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('gwp.pkl', 'rb'))

# Manually matched encodings used during training
department_map = {'finishing': 0, 'sweing': 1}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/pred", methods=["POST"])
def pred():
    quarter = int(request.form['quarter'])
    day = int(request.form['day'])
    department = department_map[request.form['department']]
    team = int(request.form['team'])
    targeted_productivity = float(request.form['targeted_productivity'])
    smv = float(request.form['smv'])
    over_time = float(request.form['over_time'])
    incentive = float(request.form['incentive'])
    idle_time = float(request.form['idle_time'])
    idle_men = int(request.form['idle_men'])
    no_of_style_change = int(request.form['no_of_style_change'])
    no_of_workers = float(request.form['no_of_workers'])
    month = int(request.form['month'])

    total = [[
        quarter, department, day, team,
        targeted_productivity, smv, over_time,
        incentive, idle_time, idle_men,
        no_of_style_change, no_of_workers, month
    ]]

    prediction = model.predict(total)[0]   # ✅ extract value

    if prediction <= 0.3:
        text = "The employee is averagely productive."
    elif prediction <= 0.8:
        text = "The employee is medium productive."
    else:
        text = "The employee is highly productive."

    #return render_template("submit.html", prediction_text=text)
    return render_template(
    "submit.html",
    label=text,
    prediction=round(float(prediction), 4)
    )



if __name__ == "__main__":
    app.run(debug=False)
