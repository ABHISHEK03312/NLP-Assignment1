import flask
from flask import Flask, request, url_for, jsonify, render_template
import pandas as pd
# from flask_cors import CORS
app = Flask(__name__)
# CORS(app)

@app.route('/', methods=['GET', 'POST'])
def search_biz():
    data = pd.read_csv("reviewSelected100.csv")
    selected_biz=request.form["Business"]
    return render_template("home.html", bid=list(data.business_id), selected_biz=selected_biz)

if __name__=="__main__":
    app.run(debug=True)

