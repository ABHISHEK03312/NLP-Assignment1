from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def getSentimentdf(df):
    sentiment = df.apply(lambda row: get_sentiment(row['text']), axis=1)
    df['sentiment'] = sentiment
    return df.sort_values(by=['sentiment'], ascending=False, ignore_index=True)
