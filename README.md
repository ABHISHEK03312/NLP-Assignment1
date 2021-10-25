# NLP-Assignment1

## Setup 

Install Dependencies for the notebook using the command
```
pip install -r requirements.txt
python -m spacy download en
```

Further run the following Commands in a .py file to download the packages:
```
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('universal_tagset')
```


### For 3.3) Indicative Adjective Phrases

Dependencies:
Stanford CoreNLP : https://stanfordnlp.github.io/CoreNLP/download.html
Prerequisites involve: 
* java 8, 
* Zip tool, 
* wget
#### Steps to download and setup:
1. Run the command: 
```
wget http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip
```

Now in the same directory where your notebook is, 
`stanford-corenlp-4.2.2` and `stanford-corenlp-4.2.2-models-english.jar` should be there
Navigate to inside the stanford-corenlp-4.2.2 directory

#### Run the command to start the server: 
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```
Note: 
The adjective phrases for each business has already been extracted using the CoreNLPParser and is stored in data/adjective_phrases/business_adjective_phrases.txt
