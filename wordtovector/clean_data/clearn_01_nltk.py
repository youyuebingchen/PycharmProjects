import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,f1_score,accuracy_score,precision_score
df = pd.read_csv("cleaned_socialmedia_relevant_cols.csv")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# 缩写还原
def replace(df,text):
    df[text] = df[text].str.replace(r"i'm","i am")
    df[text] = df[text].str.replace(r"we're","we are")
    df[text] = df[text].str.replace(r"you're","you are")
    df[text] = df[text].str.replace(r"it's", "it is")
    df[text] = df[text].str.replace(r"they're", "they are")
    df[text] = df[text].str.replace(r"he's", "he is")
    df[text] = df[text].str.replace(r"she's", "she is")
    df[text] = df[text].str.replace(r"there's","there is")
    df[text] = df[text].str.replace(r"there're","there are")
    df[text] = df[text].str.replace(r"wan't","want not")
    df[text] = df[text].str.replace(r"what's","what is")
replace(df,"text")

# 分词
df["token"] = df.text.apply(nltk.word_tokenize)

# 词形还原
def stem(sentence):
    porter = nltk.PorterStemmer()
    a = []
    for w in sentence:
        b = porter.stem(w)
        a.append(b)
    return a
df["token"] = df.token.apply(stem)

# 词根提取
def lemmer(sentence):
    porter = nltk.stem.WordNetLemmatizer()
    a = []
    for w in sentence:
        b = porter.lemmatize(w)
        a.append(b)
    return a
df["token"] = df.token.apply(lemmer)
# # print(df["token"])
# # 去停用词
def stop(text):
    stop = stopwords.words("english")
    a = [w for w in text if w not in stop]
    return  " ".join(a)
# # print(all_words)
# # all_words = [w for token in df["token"] for w in token]
# # corpus = sorted(list(set(all_words)))
# # corpus.remove("!")
# # corpus.remove("'")
# # corpus.remove("'a")
# # corpus.remove("'an")
# # corpus.remove("'am")
# # corpus.remove("'all")
# # corpus.remove("''")
# # corpus.remove("'and")
df["token"] = df.token.apply(stop)
sentence = df["token"].tolist()
# print(sentence)
# print(df['token'].tolist())
label = df["class_label"].tolist()

x_train,x_test,y_train,y_test = train_test_split(sentence,label,test_size=0.2,random_state=40)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# one hot
# def cv(data):
#     counter = CountVectorizer()
#     emb = counter.fit_transform(data)
#     return emb,counter
# x_train,counter = cv(x_train)
# x_test = counter.transform(x_test)
# the accuracy is: 0.760129;the precision is 0.759940; the recall is 0.760129';the f1 is: 0.760026

# tf-idf
def tfidf(data):
    tfidf_vector = TfidfVectorizer()
    train = tfidf_vector.fit_transform(data)
    return train,tfidf_vector
x_train,tfidf_vector = tfidf(x_train)
x_test = tfidf_vector.transform(x_test)
print(x_train)
# the accuracy is: 0.776703;the precision is 0.776577; the recall is 0.776703';the f1 is: 0.776636
# for i in sentence:
#     if len(i) <=3:
#         print(i)
# a = [len(x) for x in sentence]
# print(sorted(a))
# print(sentence)

# word2vector = Word2Vec(sentences=sentence,window=5,size=300,sample=1e-3)
# word2vector.save("disaster")
# word2vector = Word2Vec.load("disaster")
# print(word2vector.similarity("man","woman"))
# print(word2vector["love"])

# 用所有词的叠加平均的方法生成句子向量
# def get_average_vector(text,w2v,k=300):
#     if len(text)<1:
#         return np.zeros(k)
#     word_vector = [w2v[w] if w in w2v else np.zeros(k) for w in text]
#     length = len(word_vector)
#     summed = np.sum(word_vector,axis=0)
#     sen_vector = np.divide(summed,length)
#     # sen_vector = summed/length
#     return sen_vector
# df["embedding"] = df["token"].apply(lambda x:get_average_vector(x,word2vector))
# # print(df["embedding"].head())
# list_vector = df["embedding"].tolist()
# list_lable = df["class_label"].tolist()
# x_train,x_test,y_train,y_test = train_test_split(list_vector,list_lable)

# model = LogisticRegression(C=1.0,class_weight="balanced",solver="newton_cg",multi_class="multinomial",n_jobs=-1,random_state=40)
# for i in range(7,30
model = LogisticRegression(C=30,class_weight="balanced",solver="newton-cg",multi_class="multinomial",n_jobs=-1,random_state=40)
model.fit(x_train,y_train)

y_pridicted = model.predict(x_test)

def get_matix(y_test,y_pridicted):
    precsion = precision_score(y_test,y_pridicted,pos_label=None,average="weighted")
    f1 = f1_score(y_test,y_pridicted,pos_label=None,average="weighted")
    recall = recall_score(y_test,y_pridicted,pos_label=None,average="weighted")
    accuracy = accuracy_score(y_test,y_pridicted)
    return precsion,f1,recall,accuracy
precsion,f1,recall,accuracy = get_matix(y_test,y_pridicted)
print("the accuracy is: %2f;the precision is %2f; the recall is %2f';the f1 is: %2f"%(accuracy,precsion,recall,f1))
cm = confusion_matrix(y_test,y_pridicted)
print(cm)