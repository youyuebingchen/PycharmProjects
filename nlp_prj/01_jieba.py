import jieba.analyse as aly

lines = open("E:2.txt").read()
print(" ".join(aly.extract_tags(lines,topK=20,withWeight=False,allowPOS=())))