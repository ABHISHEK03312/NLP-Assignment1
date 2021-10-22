import flask
from flask import Flask, request, url_for, jsonify, render_template
import pandas as pd
# from flask_cors import CORS
app = Flask(__name__)
# CORS(app)

def summarise():
    summary="This is a dummy summary, Replace with the extracter function"
    return summary

def avg_rating(df, biz):
    biz_df=df[df['business_id']==biz]
    if len(biz_df!=0):
        return sum(biz_df['stars'])/len(biz_df)




@app.route('/', methods=['GET', 'POST'])
def search_biz():
    data = pd.read_csv("reviewSelected100.csv")
    selected_biz=request.form.get("Business")
    summary =summarise()
    rating=avg_rating(data, selected_biz)
    return render_template("home.html", bid=list(data.business_id.unique()), selected_biz=selected_biz, summary=summary, avg_rating=rating)


if __name__=="__main__":
    app.run(debug=True)

