#!/usr/bin/python
# -*- coding: UTF-8 -*-
from MultiSentiment import SentimentAnalysis

if __name__ == "__main__":

    filepath = "data/dictionary"
    obj = SentimentAnalysis()

    if obj.sentiwords is None:
        obj.loadDictionary(filepath)

    while (True):
        sen = input("请输入一句话")
        print(sen)

        wordList = obj.dataPreprocessing(sen)
        print("分词结果：{}".format(wordList))

        sentimentList = obj.computeSentiment(wordList=wordList)

        result = obj.sentimentClassify(sentimentList)

        print("情感类别：{}".format(result))

        
