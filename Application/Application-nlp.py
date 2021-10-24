# import necessary libraries
from nltk.data import retrieve
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt # we only need pyplot
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json
import transformers  # for abstractive summarization
import re
import time
import csv
import spacy
# initialize extractive summarization
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

luhn_summarizer=LuhnSummarizer()  # initializing extractive summarizer
# initializing abstractive summarizer
summarizer = transformers.TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

data = pd.read_csv("/Users/abhishekvaidyanathan/Desktop/NLP-Assignment1/data/dataset/reviewSelected100.csv")

# remove stop words
def remove_stopwords(tokenized_sentence):
    stop_words = set(stopwords.words('english'))
    stop_words = list(stop_words)
    filtered_sentence = []
    for w in tokenized_sentence:
        if w.lower() not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

# get NER tags generated from BERT
def get_ner_tags(sentence_array):
    ner_tags = []
    for sentence in sentence_array:
        ner  = nlp(sentence)
        ner_copy = ner.copy()
        i = 1
        while i < len(ner_copy):
            if(ner[i]['word'][:2]=="##" and ner[i-1]['entity']==ner[i]['entity']):
                ner_copy[i-1]['word'] = ner[i-1]['word'] + ner[i]['word'][2:]
                ner_copy[i-1]['score'] = (ner[i-1]['score'] + ner[i]['score'])/2
                ner_copy[i-1]['entity'] = ner[i-1]['entity']
                ner_copy[i-1]['index'] = ner[i-1]['index']
                ner_copy[i-1]['start'] = ner[i-1]['start']
                ner_copy[i-1]['end'] = ner[i]['end']
                ner_copy.remove(ner_copy[i])
                continue
            i = i+1
        ner_tags.append(ner_copy)
    return ner_tags

# lsa summarizer
def summarizer_lsa(text):
    parser=PlaintextParser.from_string(text,Tokenizer('english'))
    lsa_summarizer=LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document,100)
    summary = []
    for i in lsa_summary:
        summary.append(str(i))
    return summary

# concatenated reviews for every business id
def reviews_unique_business(data):
    combined_reviews = data.groupby(['business_id'], as_index = False).agg({'text': '. '.join})
    combined_reviews.text = combined_reviews.text.apply(lambda x: x.replace('\n', ''))
    combined_reviews.text = combined_reviews.text.apply(lambda x: x.replace('\r', ''))

    return combined_reviews

# get ner tags for every review. 
def bert_ner(combined_reviews):
    combined_reviews['summary'] = combined_reviews.apply(lambda row: summarizer_lsa(row['text']),axis=1)
    combined_reviews['bert_ner'] = combined_reviews.apply(lambda row: get_ner_tags(row['summary']),axis=1)

    for i in range(combined_reviews.shape[0]):
        ner_tags = combined_reviews.iloc[i]['bert_ner']
        while [] in ner_tags:
            ner_tags.remove([])
        combined_reviews.at[row,'bert_ner']= ner_tags

    return combined_reviews

combined_reviews = reviews_unique_business(data)
combined_reviews = bert_ner(combined_reviews)

# remove ner tags which are stop words or single letter words or with more than two
# consecutive repeating characters
def remove_key_value_pair(key):
    stop_words = set(stopwords.words('english'))
    stop_words = list(stop_words)
    if (key.lower() not in stop_words ) and (key.lower().isalpha()) and (len(key)>1):
        return True
    return False

# clean ner tags 
def get_cleaned_ner_tags(business_review_id_copy):
    business_review_id_clean = {}
    for key in business_review_id_copy:
        business_review_id_clean[key] = {}
        for keys in business_review_id_copy[key]:
            keys_new = re.sub("(.)\\1{2,}", "\\1", keys)
            if(remove_key_value_pair(keys_new) and business_review_id_copy[key][keys] != []):
                business_review_id_clean[key][keys_new] = business_review_id_copy[key][keys] 
    return business_review_id_clean

# get the review ids for every ner tag for a particular business
def get_business_review_id(combined_reviews):
    business_id = {}
    for i in range(combined_reviews.shape[0]):
        word = []
        for j in range(len(combined_reviews.iloc[i]['bert_ner'])):
            for k in range(len(combined_reviews.iloc[i]['bert_ner'][j])):
                word.append(combined_reviews.iloc[i]['bert_ner'][j][k]['word'])
        
        business_id[combined_reviews.iloc[i]['business_id']] = word

    for business in business_id:
        j = 0
        while j < len(business_id[business]):
            if(j>0 and business_id[business][j][:2] == '##'):
                business_id[business][j-1] = business_id[business][j-1] + business_id[business][j][2:]
                business_id[business].remove(business_id[business][j])
                continue
            j = j+1

    business_review_id = {}
    for business in business_id:
        word_dict = {}
        for word in business_id[business]:
            review_id = []
            for i in range(data.shape[0]):
                if(word in data.iloc[i]['text'].split()):
                    review_id.append(data.iloc[i]['review_id'])
            word_dict[word] = review_id
        business_review_id[business] = word_dict

    business_review_id_clean = get_cleaned_ner_tags(business_review_id)

    return business_review_id_clean

business_review_id_clean = get_business_review_id(combined_reviews)

# get the top frequent nouns
def get_most_common_nouns(text_array):
    text = ' '.join(text_array)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = remove_stopwords(tokens)
    tagged_tokens = nltk.pos_tag(tokens)
    nouns_and_verbs = []
    for token in tagged_tokens:
        if(token[1] in ['NN','NNP','NNS','NNPS'] and token[0].isalpha()):
            nouns_and_verbs.append(token[0])
    frequency = nltk.FreqDist(nouns_and_verbs)
    return frequency.most_common(10)

# get occureances in reviews for top frequent nouns
def get_reviews_top_frequent_nouns(combined_reviews,data):
    business_review_noun = {}
    for i in range(combined_reviews.shape[0]):
        word_dict_noun = {}
        for j in combined_reviews.iloc[i]['most_common_nouns']:
            review_id_noun = []
            for k in range(data.shape[0]):
                if(j in data.iloc[k]['text'].split()):
                    review_id_noun.append(data.iloc[k]['review_id'])
            word_dict_noun[j] = review_id_noun
        business_review_noun[combined_reviews.iloc[i]['business_id']] = word_dict_noun

    return business_review_noun

combined_reviews['most_common_nouns'] = combined_reviews.apply(lambda row: get_most_common_nouns(row['summary']),axis=1)
business_review_noun = get_business_review_id(combined_reviews,data)

# save files to json
def json_write(business_review_id_clean):
    with open('/Users/abhishekvaidyanathan/Desktop/NLP-Assignment1/business_review_id_clean.json', 'w') as convert_file:
        convert_file.write(json.dumps(business_review_id_clean))

    with open('/Users/abhishekvaidyanathan/Desktop/NLP-Assignment1/business_review_id_noun.json', 'w') as convert_file:
        convert_file.write(json.dumps(business_review_noun)) 


# summarization 
review_df = data.copy()
grouped = review_df.groupby('business_id')
business_ids, reviews = [], []
for name, group in grouped:
    business_ids.append(name)
    reviews.append(group.reset_index(drop=True))

def luhn_summarization():
    for i in range(len(business_ids)):
        biz_id = business_ids[i]
        temp_df = reviews[i]
        temp_revs = '\n\n'.join([s for s in temp_df['text']])
        temp_revs = re.sub('(\.)\.+', ',', temp_revs)
        
        
        start = time.time()
        print("Starting for business id:", biz_id)
        
        # extractive summarization
        temp_parser = PlaintextParser.from_string(temp_revs, Tokenizer('english'))
        temp_luhn_summary=luhn_summarizer(temp_parser.document,sentences_count=32)
        
        # greedily choosing sentences such that total tokens is less than 512
        temp_luhn_text = []
        wordlen = 0
        for sentence in temp_luhn_summary:
            s = str(sentence)
            w = len(s.split())
            if wordlen + w < 450:  # some buffer for sub-word tokenization and punctuation
                temp_luhn_text.append(s)
                wordlen += w
        temp_luhn_text = ' '.join(temp_luhn_text)
        
        # abstractive summarizer model
        temp_inputs = tokenizer("summarize: " + temp_luhn_text, return_tensors="tf", max_length=512)
        temp_outputs = summarizer.generate(temp_inputs["input_ids"], min_length=100, max_length=250, 
                                        eos_token_id=None, length_penalty=2.0, num_beams=6, 
                                        early_stopping=True, repetition_penalty=2.0)
        temp_summary = tokenizer.decode(temp_outputs[0], skip_special_tokens=True)
        end = time.time()
        print("Done! Time taken: {0:.3f} s".format(end-start))
        print("Generated summary:")
        print(temp_summary)
        
        # saving output
        with open('generated_reviews.csv', 'a') as fileobj:
            writer = csv.writer(fileobj)
            writer.writerow([biz_id, temp_summary])
        
        print("\nSummary saved successfully.\n\n")

# Additional 
summary_df = pd.read_csv('generated_reviews.csv')
summary_df.head()

nlp = spacy.load('en_core_web_sm')

def add_capitalization(inpt):
    """
    Need a function to do this because nltk and .capitalize() reduce everything to lowercase, want to preserve original upper cases like in acronyms
    """
    doc = nlp(inpt)
    tagged_sent = [(w.text, w.tag_) for w in doc]
    capitalized = []
    for i in range(len(tagged_sent)):
        w, t = tagged_sent[i]
        if t in ["NNP", "NNPS"] or w=='i' or i==0 or (i>0 and tagged_sent[i-1][0] in '.?!;'):
            capitalized.append(w[0].upper() + w[1:])  # not using .capitalize() because other capitalization is lost
        else:
            capitalized.append(w)
    res = re.sub(" (?=[\.,'!?:;])", "", ' '.join(capitalized))
    return res

add_capitalization(summary_df['summary'][0])

summary_df['summary'] = summary_df.apply(lambda x: add_capitalization(x['summary']), axis=1)
summary_df.to_csv('cleaned_reviews.csv', index=False)