import flask
from flask import Flask, request, url_for, jsonify, render_template
import pandas as pd
# from flask_cors import CORS
app = Flask(__name__)
# CORS(app)
from utils import *
import json

def summarise(biz):
    # summary=extract_summary(biz)
    data = pd.read_csv("cleaned_reviews.csv")
    for i in range(len(data)):
        if data["business_id"][i]==biz:
            return data["summary"][i]
    return "No summary found, try another business"

def avg_rating(df, biz):
    biz_df=df[df['business_id']==biz]
    if len(biz_df!=0):
        return sum(biz_df['stars'])/len(biz_df)
    return 0

def avg_senti(df, biz):
    biz_df=df[df['business_id']==biz]
    biz_df=getSentimentdf(biz_df)
    if len(biz_df!=0):
        return sum(biz_df['sentiment'])/len(biz_df)
    return 0

def top_positive(df, biz):
    biz_df=df[df['business_id']==biz]
    biz_df=getSentimentdf(biz_df)
    print(biz_df)
    return biz_df['text'][0]

def top_negative(df, biz):
    biz_df=df[df['business_id']==biz]
    biz_df=getSentimentdf(biz_df)
    return biz_df['text'][len(biz_df)-1]

def ReadNER(biz):
    # NER_dict={"ZBE-H_aUlicix_9vUGQPIQ":{"NER1":["r1", "r2"]}}
    f = open ('business_review_id_new.json', "r")
    NER_dict = json.loads(f.read())
    return NER_dict[biz]
    # return {"NER1":["r1", "r2"], "NER2":["r3", "r4"]}

data_dict={
    "bid":[],
    "selected_biz":"",
    "summary":"Select a Business to learn more",
    "rating":0,
    "ners":[],
    "nerData":[],
    "senti":0.0,
    "pos":"",
    "neg":""}

@app.route('/', methods=['GET', 'POST'])
def search_biz():
    data = pd.read_csv("reviewSelected100.csv")
    selected_biz=request.form.get("Business")
    summary =summarise(selected_biz)
    rating=avg_rating(data, selected_biz)
    if selected_biz==None:
        nerDict={}
    else:
        # print(selected_biz)
        nerDict=ReadNER(selected_biz)
        data_dict["senti"]=avg_senti(data, selected_biz)
        data_dict["pos"]=top_positive(data, selected_biz)
        data_dict["neg"]=top_negative(data, selected_biz)
    data_dict["bid"]=list(data.business_id.unique())
    data_dict["summary"]=summary
    data_dict["selected_biz"]=selected_biz
    data_dict["rating"]=rating
    data_dict["ners"]=nerDict.keys()
    
    selected_ner=request.form.get("nerInfo")
    nerDict=data_dict["selected_biz"]
    if selected_ner==None:
        nerDet=[]
    else:
        nerDet=nerDict[selected_ner]
    return render_template("home.html", bid=data_dict["bid"], selected_biz=data_dict["selected_biz"], summary=data_dict["summary"], avg_rating=data_dict["rating"],  ners=data_dict["ners"], nerDet=data_dict["nerData"], senti=data_dict["senti"], pos=data_dict["pos"], neg=data_dict["neg"])

@app.route('/<nerInfo>', methods=['GET', 'POST'])
def search_ner(nerInfo=None):
    # if selected_ner==None:
    #     nerDet=[]
    # else:
    #     nerDet=nerDict[selected_ner]
    return render_template("home.html", bid=data_dict["bid"], selected_biz=data_dict["selected_biz"], summary=data_dict["summary"], avg_rating=data_dict["rating"], ners=data_dict["ners"], nerDet=data_dict["nerData"])
# @app.route('/', methods=['GET', 'POST'])
# def ner_biz():
#     data = pd.read_csv("reviewSelected100.csv")
#     selected_biz=request.form.get("Business")
#     summary =summarise(selected_biz)
#     rating=avg_rating(data, selected_biz)
#     return render_template("home.html", bid=list(data.business_id.unique()), selected_biz=selected_biz, summary=summary, avg_rating=rating)
if __name__=="__main__":
    app.run(debug=True)

