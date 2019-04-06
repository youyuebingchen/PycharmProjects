# nltk 使用手册
## 词频提取

- all_words= nltk.FreqDist(corpus)
- all_words.keys()
- all_words.plot() 



## 英文分词
### 缩写还原
- df[text]=df[text].str.replace(r"i'm","i am")
### 分词
- text = nltk.word_tokenize("text") 返回list
### 分句
- text = nltk.sentence_tokenize("text")

## 词干提取
### 词根还原：不同形式还原
- porter = nltk.PorterStemmer()
- porter.stem("lying")
### 词形还原：提取词根
- porter = nltk.stem.WordNetLemmatizer()
- porter.lemmatize("word")

## 词性标注
能根据不同的语境对单词进行词性标注
- nltk.pos_tag(["i","love","you"])
### 词性标注语料制作
- tagged_token = nltk.tag.str2tuple("fly/NN")
### 词性标注器
- default_tagger = nltk.DefaultTagger('NN')
- tags = default_tagger.tag(tokens)
