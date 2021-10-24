from flask import Flask, request, url_for, jsonify, render_template
import pandas as pd
# from flask_cors import CORS
app = Flask(__name__)
# CORS(app)
from utils import *

@app.route('/', methods=['GET', 'POST'])
def search_biz():

    data = pd.read_csv("reviewSelected100.csv")
    # default values
    data_dict={
        "bid":list(data.business_id.unique()),
        "selected_biz":"",
        "summary":"Select a Business to learn more",
        "avg_rating":0,
        "senti":0.0,
        "pos":"",
        "neg":"",
        "ners":[],
        "selected_ner":"",
        "nerData":[],
        "flag1":False,
        "flag2":False
    }

    if request.method == 'GET':
        return render_template("home.html", **data_dict)

    elif request.method == 'POST':
        selected_biz=request.form.get("Business")
        print("Selected business:", selected_biz)
        if selected_biz == "":  # submission made without selecting business, just return default page
            return render_template("home.html", **data_dict)
        
        nerDict=ReadNER(selected_biz)

        data_dict["selected_biz"]=selected_biz
        data_dict["summary"]=summarise(selected_biz)
        data_dict["avg_rating"]=avg_rating(data, selected_biz)
        data_dict["senti"]=round(avg_senti(data, selected_biz)*2.5 + 2.5, 2)  # putting sentiment on a 0-5 scale
        data_dict["pos"]=top_positive(data, selected_biz)
        data_dict["neg"]=top_negative(data, selected_biz)
        data_dict["ners"]=nerDict.keys()
        data_dict["flag1"] = True
        
        selected_ner=request.form.get("nerInfo")
        print("Selected NER:", selected_ner)
        if selected_ner!=None:
            data_dict["selected_ner"] = selected_ner
            data_dict["nerData"] = get_NER_reviews(nerDict[selected_ner], data)
            data_dict["flag2"] = True
        
        return render_template("home.html", **data_dict)


if __name__=="__main__":
    app.run(debug=True)

