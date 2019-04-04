import pandas as pd

pd.set_option("display.max_columns",None)
df = pd.read_csv("E:\socialmedia_relevant_cols.csv",encoding="ISO-8859-1")
df = df.rename(columns={"choose_one":"choose"})
# print(df.query('class_label==["2"]'))

# 删除class_label == 2 的数据
df = df[(True^df["class_label"].isin([2]))]
# print(df["text"][0:5])
# df = df.sample(n=3000)
# print(df.describe().round(2).T)
# print(df.groupby("class_label").count())
# print(df.groupby(["class_label","choose"])["text"].count())
# 数据清洗
# 去除所有的http//:
df["text"] = df["text"].str.replace(r"http\S+","")
df["text"] = df["text"].str.replace(r"http","")
df["text"] = df["text"].str.replace(r"@\S+","")
df["text"] = df["text"].str.replace(r"@","at")
#去除非a-zA-Z0-9的所有符号
df["text"] = df["text"].str.replace("[^a-zA-Z,?!.;'""]"," ")
# 大写转化成小写
df["text"] = df["text"].str.lower()
# 保存数据
df.to_csv("cleaned_socialmedia_relevant_cols.csv")
print(df["text"][0])
# print(df["text"].str.contains("@"))
