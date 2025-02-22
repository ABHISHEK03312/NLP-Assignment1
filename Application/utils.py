from textblob import TextBlob
import json
import pandas as pd

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def getSentimentdf(inp_df):
    df = inp_df.copy()
    df['sentiment'] = df.apply(lambda row: get_sentiment(row['text']), axis=1)
    return df.sort_values(by=['sentiment'], ascending=False, ignore_index=True)

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
    # print(biz_df)
    return biz_df['text'][0]

def top_negative(df, biz):
    biz_df=df[df['business_id']==biz]
    biz_df=getSentimentdf(biz_df)
    return biz_df['text'][len(biz_df)-1]

def ReadNER(biz):
    # NER_dict={"ZBE-H_aUlicix_9vUGQPIQ":{"NER1":["r1", "r2"]}}
    with open('business_review_noun.json', "r") as f:
        NER_dict = json.loads(f.read())
    return NER_dict[biz]
    # return {"NER1":["r1", "r2"], "NER2":["r3", "r4"]}

def read_POS_NER(biz_id):
    with open('pos_ner_reviews.json', "r") as f:
        business_ner_reviews = json.loads(f.read())
    with open('pos_ner_counts.json', "r") as f:
        business_ner_counts = json.loads(f.read())
    topNERs = []
    for k in business_ner_reviews[biz_id].keys():
        topNERs.append((k, business_ner_counts[biz_id][k]))
    topNERs.sort(key=lambda pair: pair[1], reverse=True)
    nerDict = {}
    for ner, occ in topNERs[:10]:
        nerDict[ner] = business_ner_reviews[biz_id][ner]
    return nerDict

def get_NER_reviews(reviewIds, data):
    reviewSet = set(reviewIds)
    reviewList = []
    for i in range(len(data)):
        if data['review_id'][i] in reviewSet:
            reviewList.append(data['text'][i])
    if not len(reviewList):
        reviewList.append("No reviews for this tag, please choose a different one.")
    return reviewList