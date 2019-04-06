# 数据清洗
# import pandas as pd
# from bs4 import BeautifulSoup
# # # 导入数据 TODO
# pd.set_option("display.max_columns",None,"display.max_colwidth",200)
# df = pd.read_csv("E:\kaggle-word2vec-movie-reviews-master\data\labeledTrainData.tsv",sep="\t",escapechar="\\")
# # 观察数据 TODO
# print(df.info())
# print(df.groupby("sentiment")["review"].count())
# print(df.describe().round(2).T)
# # 清洗数据
# # 去除网络标签
# df["review"] = df.review.apply(lambda x:BeautifulSoup(x,"html.parser").get_text())
# # print(df["review"].head())
# # 去除非a-zA-Z0-9？！，。等符号
# df["review"] = df["review"].str.replace(r"[^a-zA-Z0-9,.'?!]"," ")
# # print(df["review"].head())
# # 大写将为小写
# df["review"] = df["review"].str.lower()
# print(df["review"].head())
# 保存数据
# df.to_csv("bagofword.csv")
import nltk
from nltk.corpus import stopwords
# nltk TODO
# 缩写还原
# df["review"] = df["review"].str.replace(r"i'm","i am")
# df["review"] = df["review"].str.replace(r"i've","i have")
# df["review"] = df["review"].str.replace(r"you're","you are")
# df["review"] = df["review"].str.replace(r"you've","you have")
# df["review"] = df["review"].str.replace(r"he's","he is")
# df["review"] = df["review"].str.replace(r"she's","she is")
# df["review"] = df["review"].str.replace(r"they're","they are")
# df["review"] = df["review"].str.replace(r"wan't","want not")
# df["review"] = df["review"].str.replace(r"don't","do not")
# df["review"] = df["review"].str.replace(r"didn't","did not")
# print(df["review"].head())
# 分词
# df["token"] = df.review.apply(nltk.word_tokenize)
# print(df["token"].head())
# # 词形还原 stem
# def stemer(text):
#     b = []
#     porter = nltk.PorterStemmer()
#     for w in text:
#         a = porter.stem(w)
#         b.append(a)
#     return b
# df["token"] = df.token.apply(stemer)
# print(df["token"].head())
# # 词根提取 lemmer
# def lemmer(text):
#     lemm = nltk.stem.WordNetLemmatizer()
#     b = []
#     for w in text:
#         a = lemm.lemmatize(w)
#         b.append(a)
#     return b
# df["token"] = df.token.apply(lemmer)
# print(df["token"].head())
# 去停用词
# stop = stopwords.words("english")
# # print(stop)
# def remove(text):
#     a = [w for w in text if w not in stop]
#     return " ".join(a)
# df["token"] = df.token.apply(remove)
# # # 保存数据
# df.to_csv("bagofword.csv")


import pandas as pd
pd.set_option("display.max_columns",None,"display.max_colwidth",200)
df = pd.read_csv("bagofword.csv")
# print(df["review"].head())
# print(df["token"].head())

# import nltk
# df["token"] = df.token.apply(nltk.word_tokenize)
list_corpus = df["token"].tolist()
list_label = df["sentiment"].tolist()
from sklearn.model_selection import train_test_split
# 进行训练集划分 TODO
x_train,x_test,y_train,y_test = train_test_split(list_corpus,list_label,test_size=0.2,random_state=1)

from sklearn.feature_extraction.text import CountVectorizer
# 模型训练 onehot TODO
# def cv(data):
#     counter = CountVectorizer()
#     emb = counter.fit_transform(data)
#     return emb,counter
# x_train,counter = cv(x_train)
# x_test =counter.transform(x_test)
# 训练模型
import numpy as npb
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# clf = LogisticRegression(penalty="l2",C=6,class_weight="balanced",n_jobs=-1,random_state=40,solver="saga")
# clf.fit(x_train,y_train)
# from sklearn.model_selection import KFold
# from sklearn.metrics import precision_score
# #  交叉验证防止过拟合
# # kf.split 得到是x_train 的index,所以下面要通过调用Index来实现。
# kf = KFold(n_splits=5,random_state=1,shuffle=False)
# precision1 = []
# for train,test in kf.split(x_train):
#     # print(train,test)
#     # clf = svm.SVC(C=1.0,class_weight="balanced",random_state=40)
#     # clf = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=1)
#     clf.fit(x_train[train[0]:train[-1]],y_train[train[0]:train[-1]])
#     y_pre = clf.predict(x_train[test[0]:test[-1]])
#     pre = precision_score(y_train[test[0]:test[-1]],y_pre)
#     print(pre)
#     # [0.8635, 0.9137104506232023 ,0.9132804757185332, 0.9244741873804971, 0.8699067255768287]
#     precision1.append(pre)
# # y_predict_train = clf.predict(x_train)
# y_predict_train = np.sum(precision1)/len(precision1)
# print(y_predict_train)
#  precision：0.8969743678598124
# y_predict = clf.predict(x_test)
# y_predict_train = clf.predict(x_train)
# kfold: precisoin:0.879846; accuracy:0.879800; recall:0.879800; f1:0.879797
# l1 c=6 solver=saga
# [[2189  314]
#  [ 282 2215]]
# precisoin:0.880864; accuracy:0.880800; recall:0.880800; f1:0.880796
# [[9183  814]
#  [ 637 9366]]
# precisoin:0.927584; accuracy:0.927450; recall:0.927450; f1:0.927444
# l2 c=6 solver=saga
# [[2192  311]
#  [ 281 2216]]
# precisoin:0.881657; accuracy:0.881600; recall:0.881600; f1:0.881597
# [[9220  777]
#  [ 610 9393]]
# precisoin:0.930770; accuracy:0.930650; recall:0.930650; f1:0.930645
# # # 模型训练 tf-idf TODO
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf(text):
    tfidf = TfidfVectorizer()
    emb = tfidf.fit_transform(text)
    return  emb,tfidf

x_train,tfidf = tfidf(x_train)
x_test = tfidf.transform(x_test)
import numpy as np
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty="l2",C=1.5,class_weight="balanced",solver="saga",random_state=40,n_jobs=-1)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
y_predict_train = clf.predict(x_train)
# l2 c=1.5 saga
# [[2209  294]
#  [ 245 2252]]
# precisoin:0.892353; accuracy:0.892200; recall:0.892200; f1:0.892191
# [[9312  685]
#  [ 515 9488]]
# precisoin:0.940127; accuracy:0.940000; recall:0.940000; f1:0.939996
# import gensim
# import numpy as np
# from gensim.models import Word2Vec
# # 模型训练 w2v TODO
# # model = Word2Vec(sentences=list_corpus,size=300,window=5,min_count=5,sample=1e-3,sg=1)
# # model.save("bag.save")
# word2vector = Word2Vec.load("bag.save")
# # print(list_corpus)
# print(word2vector.most_similar("hate"))
# # def avetage(text,size=300):
# #     if len(text) <1:
# #         return np.zeros(size)
# #     a = [word2vector[w] if w in word2vector else np.zeros(size) for w in text]
# #     length = len(a)
# #     summed = np.sum(a,axis=0)
# #     ave = np.divide(summed,length)
# #     return ave
# # df["token"] = df.token.apply(avetage)
# # list_corpus = df["token"].tolist()
# # x_train,x_test,y_train,y_test = train_test_split(list_corpus,list_label,random_state=1,test_size=0.2)
# #
# # from sklearn.linear_model import  LogisticRegression
# # clf = LogisticRegression(penalty="l2",C=1.0,solver="newton-cg",class_weight="balanced",n_jobs=-1)
# # clf.fit(x_train,y_train)
# # y_predict = clf.predict(x_test)
# # # [[1482 1021]
# # #  [ 937 1560]]
# # # precisoin:0.608538; accuracy:0.608400; recall:0.608400; f1:0.608297
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix

# 模型评估 confuse matrix accuracy recall presicion f1
def score_matrix(y_test,y_predicted):
    precision = precision_score(y_test,y_predicted,pos_label=None,average="weighted")
    accuracy = accuracy_score(y_test,y_predicted)
    recall = recall_score(y_test,y_predicted,pos_label=None,average="weighted")
    f1 = f1_score(y_test,y_predicted,pos_label=None,average="weighted")
    return precision,accuracy,recall,f1
precision,accuracy,recall,f1 = score_matrix(y_test,y_predict)
cm = confusion_matrix(y_test,y_predict)
print(cm)
print("precisoin:%2f; accuracy:%2f; recall:%2f; f1:%2f"%(precision,accuracy,recall,f1))
precision,accuracy,recall,f1 = score_matrix(y_train,y_predict_train)
cm = confusion_matrix(y_train,y_predict_train)
print(cm)
print("precisoin:%2f; accuracy:%2f; recall:%2f; f1:%2f"%(precision,accuracy,recall,f1))