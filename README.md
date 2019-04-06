# NLP 一般流程
## 收集数据、准备、检查数据
### 第一步：收集数据
注意：labels

- Richard:rather than spending a month foguring out an unsurpervised mathine learning problem, just label some data for a week and train a classifier.
### 第二步：清洗数据
原则：再好的模型也拯救不了shi一样的数据。

#### 一、导入数据

- pd.set_option("display.max_columns",None,"display.max_colwidth",200)
- df = pd.read_csv("")
#### 二、观察数据：

- df.shape
- df.info()
- df.describe().round(2).T
- df.dtypes
- df["B"].dtype
- df.isnull()
- df["B"].isnull
- df["B"].unique()
- df.values
- df.columns
- df.head()
- df.tail()

#### 三、数据清洗

- df.fillna(value = 0)
- df["B"].fillna(df["B"].mean())
- df["B"] = df["B"].map(str.strip) 清楚B字段的字符空格
- df["B"] = df["B"].str.lower() 大小写转换
- df["B"].astype("int")更改数据格式： 
- df = df.rename(columns = {"B"："A"})更改列名
- df["B"].frop_duplicate(keep="last") 删除重复出现的值
- df["B"].replace("sh","shanhai") 数据替换]
- df = df[(True^df["class_label"].isin([2]))

#### 四、数据汇总

主要使用groupby 和pivote_table

- df.groupby("city").count() 对所有列进行数据汇总
- df.groupby("city")["id"].count() 按城市对id 字段进行计数
- df.groupby(["city","size"])["id"].count()
- df_inner.groupby('city')['price'].agg([len,np.sum, np.mean])

#### 五、文本清洗
 
##### 1、去除网络标签
- df["revivew"] = df.review.apply(lambda x:BeautifulSoup(x,"html.parser).get_text())
##### 2、删除所有不相关的字符，如任何非字母、数字字符
- df["text"].str.replace(r"[^A-Za-z0-9,.!'?]")
##### 3、删除不相关的字词例如 @...
- df["text"].str.replace(r"@","at")
##### 4、将大写变为小写
- df["text"].str.lower()
##### 5、考虑整合多种拼写错误或者多种拼写的单词
##### 6、缩写还原
- df["text"]=df["text"].str.replace(r"i'm","i am")
##### 7、分词
- df["token"] = df.review.apply(nltk.word_tokenize)
##### 8、词形还原stem
- def stemmer(text): 
- b = [];porter = nltk.PorterStermmer()
- for w in text:
- a = porter.stem(w);b.append(a)
- return b
- df["token"] = df.token.apply(stemer)
##### 9、词根提取 lemmer
- def lemmer(text): 
- b = [];porter = nltk.stem.WordNetLemmatizer()
- for w in text: 
- a = porter.lemmatize(w);b.append(a)
- return b
- df["token"] = df.token.apply(lemmer)
##### 10、去停用词
- stop = stopwords.words("english")
- def remove(text):
- a = [w for w in text if w not in stop]
- return " ".join(a)
- df["token"] = df.token.apply(remove)


#### 六、数据预处理

##### 1、数据合并

- df_inner = pd.merge(df,df1,how="inner")交集
- df_lef = pd.merge(df,df1,how = "left") 左 
- df_right = pd.merge(df.df1,how="right")右
- df_outer = pd.merge(df,df1,how="outer) 并

##### 2、设置索引列

- df_inner.set_index("id")

##### 3、按特定列的值排序

- df_inner.sort_values(by=["age"])

#### 七、数据提取

##### 1、按索引提取单行的数值 loc iloc ix

- df.loc[3]
- df.iloc[:3,:2]
- df.iloc[[0,2,5],[4,5]]
- df.reset_index()
- de.set_index("date)
- df["B"].isin(["beijing"]) 判断北京是否在B列
- pd.DataFrame(category.str[:3]) 提取前三个字符并生成数据表

#### 八、数据筛选

##### 1、使用与或非配合大于小于等于对数据进行帅选并计数求和。

- df.query('city == ["bei jing"]').price.sum()

#### 九、数据统计

- 简单的数据采样： df.sample(n=3)
- 手动设置权重： weights = [0,0.1,0.2], df.sample(n=2,weights=weights)
- 采样不放回： df.sanple(n=6,replace=False)
- 数据统计描述 df.describe().round(2).T  round 设置显示小数位 
- 计算标准差： df["B"].std()
- 计算协方差：df["B"].cov(df["A"])
- df.cov()
- 计算两个字段的相关性分析： df["B"].corr(df["A"])
- df.corr()

#### 十、数据输出

- df.to_excel("a.xlsx")
- df.to_csv("a.csv")

### 第三步：找到一个好的数据表示方式
#### 划分数据集
- list_corpus = df["text"].tolist()
- list_label = df["label"].tolist()
- from sklearn.model_selection import train_test_split
- x_train,x_test,y_train,y_test = train_test_split(list_corpus,list_label)
#### 一、one hot 独热编码(可视化嵌入)
- from sklearn.feature_extraction.text import CountVectorizer
- def cv(text):
- counter = CounterVectorizer()
- emb = counter.fit_trainsform(text)
- return emb,counter
- x_train,counter = cv(x_train)
- x_test = counter.transform(text)

#### 二、Tf-idf 
- from sklearn.feature_extraction.text import TfidfVectorizer
- def tfidf(text):
- tfidf = TfidfVectorizer()
- emb = tfidf.fit_transform(text)
- return emb,tfidf
- x_train.counter = tfidf(x_train)
- x_test = counter.transform(text)

#### 三、word2vector
##### 1、训练 model
- import gesim
- import numpy as np
- from gesim.models import Word2Vec
- model = Word2Vec(sentences=list_corpus,size=300,window=5,min_count=5,sample=1e-3,sg=1)
- model.save("bag") 注意：此处sentence 是分词过后的。例子：[["i","love","you"],["do","you","love","me"]]
##### 2、将句子转化成向量
用平均的方法将句子转换成向量 再进行训练集划分

- word2vector = Word2Vec.load("bag")
- def average(text,size=300)
- if len(text) < 1:
- return np.zeros(size)
- a = [word2vector[w] if w in word2vector else np.zeros(size) for w in text]
- length = len(a)
- summed = np.sum(a,axis=0)
- ave = np.divide(summed,length)
- return ave
- df["text"] = df["text"].apply(average) 注意此处的df["text"] 未分词
- list_corpus = df["text"].tolist()
- list_label = df["label"].tolist()
- x \_train,x\_test,y\_train,y\_test = trian\_test\_split(list\_corpus,list\_label,test\_size=0.2,random_state=1.0)


## 建立简单的模型
### 第四步：分类
- from sklean.linear_model import LogisticRegression
- clf = LogisticRegression(penalty="l2",C=1.0,class_weight="balanced",n_jobs=-1,random_state=1.0,solver="newton-cg")
- clf.fit(x_train,y_train)
- y_predict = clf.fit(x_test)

## 理解解释模型
### 第五步：检查
#### 混淆矩阵
- from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
- from sklearn.metrics import confusion_matrix
- precision = precision_score(y_test,y_predic,pos_label=None,average="weighted")
- accuracy = accuracy_score(y_test,y_predict)
- recall = recall_score(y_test,y_predict,,pos_label=None,average="weighted")
- f1 = f1_score(y_test,y_predict,pos_label,average="weighted")
- cm = confusion_matrix(y_test,y_predict)
#### 模型分析
### 第六步：统计词结构
TF-IDF：关键词、可视化嵌入
### 第七步：巧妙利用语义
#### 将词转化为向量
#### 使用预训练的词
#### 句级别的表示
#### w2v 句嵌入
#### 复杂性和可解释性的权衡
### 第八步：使用端对端的方法来巧妙利用语义
#### 将句子作为一个词向量序列
word2vec、glove、cove 
