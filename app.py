#Use this file to preprocess new data and predict using the saved model or pickel file
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from nltk import ngrams
from nltk.tokenize import word_tokenize
import nltk
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
from collections import Counter
import math
import textstat
import sys
from textblob import TextBlob
from empath import Empath
lexicon = Empath()
def unique_words(text):
    return len(set(text.split()))
def lex(x):
    lx = lexicon.analyze(x, normalize=True)
    return pd.Series((lx['positive_emotion'],lx['negative_emotion']))
def reading_ease(text):
    return textstat.flesch_reading_ease(text)
def detectextreme(a):
	if a == 0.0 or a== 10.0:
		return 1
	else:
		return 0
def Counter1(text):
    return(Counter(text.split()))
def POS_Tagging_Nouns(sentence):
    tagged_list = []
    tags = []
    count_verbs = 0
    count_nouns = 0
    text=nltk.word_tokenize(sentence)
    tagged_list = (nltk.pos_tag(text))
    tags = [x[1] for x in tagged_list]
    for each_item in tags:
        if each_item in ['VERB','VB','VBN','VBD','VBZ','VBG','VBP']:
            count_verbs+=1
        elif each_item in ['NOUN','NNP','NN','NUM','NNS','NP','NNPS']:
            count_nouns+=1
    return count_nouns

def POS_Tagging_Verbs(sentence):
    tagged_list = []
    tags = []
    count_verbs = 0
    count_nouns = 0
    text=nltk.word_tokenize(sentence)
    tagged_list = (nltk.pos_tag(text))
    
    tags = [x[1] for x in tagged_list]
    for each_item in tags:
        if each_item in ['VERB','VB','VBN','VBD','VBZ','VBG','VBP']:
            count_verbs+=1
        elif each_item in ['NOUN','NNP','NN','NUM','NNS','NP','NNPS']:
            count_nouns+=1
    return count_verbs
def cosine_similarity_ngrams(a, b):
    vec1 = Counter(a)
    vec2 = Counter(b)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	data = pd.read_csv('data/cosinedupedata1.csv')
	
	df = data[['cosine_sim', 'No_Of_Unique_Words', 'deviation', 'positive_emotion', 'negative_emotion', 'reading_ease' ]]
	cols = df.columns
	scaler = MinMaxScaler(feature_range=(0, 1))

	rescaledX = scaler.fit_transform(df)
	df = pd.DataFrame(rescaledX, columns=cols)
	model = KMeans(2)
	model.fit(df)
	#clust_labels = model.predict(df)
	#Alternative Usage of Saved Model
	#ytb_model = open("kmeansmodel.pckl","rb")  #If using .pckl file use this to load model
	#model = joblib.load(ytb_model)

	if request.method == 'POST':
		test = pd.DataFrame(columns=['date', 'QuantitativeScore_x', 'SentimentText', 'SiteId' ])
		comment = request.form['comment']
		ldate = request.form['rdate']
		test = test.append({'date': ldate, 'QuantitativeScore_x': int(request.form['rating']), 'SentimentText':request.form['comment'], 'SiteId': request.form['SiteId'], 'PropertyId':8250}, ignore_index=True)
		test1 = data[(data['PropertyId'] == 8250) & (data['date']== request.form['rdate'] )]
		test1['SentimentText']=test1['SentimentText'].fillna("")
		test1['bigram']=test1['bigram'].fillna("")
		test['SentimentText']=test['SentimentText'].fillna("")
		test['text1'] = test['SentimentText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
		test['text1'] = test['text1'].str.replace('[^\w\s]','')
		test['bigram'] = test['text1'].apply(lambda row: list(ngrams(word_tokenize(row), 2)))
		csf = 0
		for index1, row1 in test1.iterrows():
					atext = test["SentimentText"].values[0]
					btext = row1["SentimentText"]
					a = ngrams(word_tokenize(atext), 2)
					b = ngrams(word_tokenize(btext), 2)
					cs = cosine_similarity_ngrams(a,b)
					#print('ReviewId', row1["bigram"], 'cs', cs)
					if csf < cs and cs != 1.0:
						csf = cs   
		test['cosine_sim'] = csf
		temp = data.loc[data['PropertyId'] == 8250]['QuantitativeScore_x'].mean()
		test['Quantitative_Score_y'] = temp
		test['deviation'] = abs(test['Quantitative_Score_y'] - test['QuantitativeScore_x'])
		test[['positive_emotion', 'negative_emotion']] = test.apply(lambda row: pd.Series(lex(row['text1'])), axis=1)
		test['No_Of_Unique_Words']= test['text1'].apply(unique_words)
		test['reading_ease'] = test['text1'].apply(reading_ease)
		'''
		#Features which are not being used in modeling
		test['extreme'] = test['QuantitativeScore_x'].apply(detectextreme)
		test['length'] = test['SentimentText'].str.len()
		test['property_date_count'] = test1['ReviewId_x'].count()
		test['POS_Tags_Nouns']=test['text1'].apply(POS_Tagging_Nouns)
		test['POS_Tags_Verbs']=test['text1'].apply(POS_Tagging_Verbs)
		test['sentiment'] = test['SentimentText'].apply(lambda x: TextBlob(x).sentiment[0])
		'''
		test_x = test[[ 'cosine_sim', 'No_Of_Unique_Words', 'deviation', 'positive_emotion', 'negative_emotion', 'reading_ease' ]]
		rescaledX = scaler.fit_transform(test_x)
		test_x = pd.DataFrame(rescaledX, columns=cols)
		#vect = cv.transform(data).toarray()
		my_prediction = model.predict(test_x)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)