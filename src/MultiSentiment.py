#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import jieba
import numpy as np
import re
import codecs
from copy import deepcopy

class SentimentAnalysis(object):
    """情感分类，基于词典和规则


       Args:
           sentiwords: 情感词典，共7大类情感
           negative: 否定词词典
           degree: 程度副词词典
           sentiment: 情感类别
           opposite: 情感之间的对立关系
           degreeStrength: 不同类别的程度副词的程度值
           sentscore: 默认情感强度
           threshold: 情感强度阈值，低于该强度情感被归到默认情感类别
           additional: 用户可以手动增减的情感词典


        Returns:
            用户语句所属情感类型
    """

    # 用户可通过调用方法来更改的词典



    def __init__(self, threshold = 2):
        self.sentiwords = None  # 初始化情感词词典
        self.negative = {}  #  初始化否定词词典
        self.degree = {}  # 初始化程度副词词典
        self.sentiment = ["快乐","喜欢","生气","悲伤","害怕","讨厌","惊讶"]
        self.opposite = {"快乐":"生气","喜欢":"讨厌","生气":"快乐","悲伤":"快乐","害怕":"快乐","讨厌":"喜欢","惊讶":"害怕"}  #  预定义情感分类的对立关系，比如 '乐'的对立面是'怒'等
        self.degreeStrength = {"extreme":3,"very":1.4,"more":2,"alittlebit":1.4,"insufficient":.8,"over":2.2}  #  定义每一类程度副词的程度值
        self.sentscore = np.zeros(len(self.sentiment))  #  默认情感强度
        self.threshold = threshold  #  定义情感强度阈值
        self.additional = {}

    # 数据预处理模块
    def dataPreprocessing(self,line):
        """
        对用户文本进行预处理，包括剔除非中文，分词
        :param line: 用户输入语句
        :return: 切分后的wordlist
        """

        line = ''.join(re.findall(r'[\u4e00-\u9fa5]',line))
        line = jieba.cut(line)
        return list(line)

    def _readtable(self,path):
        """
        将输入的txt文本村放入list
        :param path: 文本相对路径
        :return:
        """

        ans = []
        with codecs.open(path, "r", "utf-8") as f:
            for line in f.readlines():
                ans.append(line.strip())
        return ans


    def _groupClass(self,category):
        """
        将情感词典中21小类转化为7大类
        :param category: 类别
        :return:
        """
        if category in ["PA", "PE"]:
            return "快乐"
        if category in ["PD", "PH", "PG", "PB", "PK"]:
            return "喜欢"
        if category in ["NA"]:
            return "生气"
        if category in ["NB", "NJ", "NH", "PF"]:
            return "悲伤"
        if category in ["NI", "NC", "NG"]:
            return "害怕"
        if category in ["NE", "ND", "NN", "NK", "NL"]:
            return "讨厌"
        if category in ["PC"]:
            return "惊讶"


    def _addDict(self,path):
        """
        在基础情感词典上添加更多情感词典
        :param path: 增添的情感词典路径
        :return:
        """

        df = pd.read_csv(path)
        df = df.set_index("词语").T.to_dict("list")
        return {key:value for key,value in df.items() if key not in self.sentiwords.keys()}


    #  词典加载模块
    def loadDictionary(self, base_path = "data/dictionary"):
        """
        加载模块情感词、否定词、程度副词词典
        对self.sentiwords, self.negative, self.degree进行赋值
        :param base_path: 词典所在的文件夹路径
        :return:
        """
        self.negative = self._readtable(base_path + "/negative/no.txt")

        alittlebit = self._readtable(base_path + "/degreewords/alittlebit.txt")
        extreme = self._readtable(base_path + "/degreewords/extreme.txt")
        insufficient = self._readtable(base_path + "/degreewords/insufficiently.txt")
        more = self._readtable(base_path + "/degreewords/more.txt")
        over = self._readtable(base_path + "/degreewords/over.txt")
        very = self._readtable(base_path + "/degreewords/very.txt")

        self.degree = {"alittlebit":alittlebit, "extreme":extreme,
                       "insufficient":insufficient, "more":more,
                       "over":over, "very":very}

        raw_sentiwords = pd.read_csv(base_path + "/sentiwords/sentiwords.csv")
        raw_sentiwords = pd.DataFrame(raw_sentiwords,columns = ["词语","情感分类","强度"])
        newClass = list(map(self._groupClass,raw_sentiwords["情感分类"]))
        raw_sentiwords["情感分类"] = newClass
        self.sentiwords = raw_sentiwords.set_index("词语").T.to_dict("list")


        #sad = self._addDict(base_path + "/sentiwords/sad.csv")
        #happy = self._addDict(base_path + "/sentiwords/happy.csv")
        #hate = self._addDict(base_path + "/sentiwords/hate.csv")
        #like = self._addDict(base_path + "/sentiwords/like.csv")
        #surprise = self._addDict(base_path + "/sentiwords/surprise.csv")
        #angry = self._addDict(base_path + "/sentiwords/angry.csv")
        #terrible = self._addDict(base_path + "/sentiwords/terrible.csv")
        #add_dict = [sad,happy,hate,like,surprise,angry,terrible]
        
        #additional = pd.read_csv(base_path + "/sentiwords/additional.csv")
        #self.sentiwords = additional.set_index("词语").T.to_dict("list")

        #for dic in add_dict:
            #self.sentiwords.update(dic)
        #self.sentiwords.update(self.additional)


    def computeSentiment(self, wordList, window_size = 3):
        """
        计算每个句子的情感强度
        :param wordList: 单词list/array
        :return: 类型：array, 记录每个类别下的情感强度
        """
        W = 1  # 初始化权重
        sentistrength = deepcopy(self.sentscore) # 初始化情感强度


        for i in range(len(wordList)):  # 循环变量单词list
            # 多个if，判断单词是否在词典中,属于哪个词典
            signal = 0  #  出现一个程度词，判断该程度词是否修饰一个情感词

            if wordList[i] in self.negative:  # 如果是否定词，需要判定向后三个单词窗口的是否有情感词
                j = i + 1
                while j < len(wordList) and j - i < window_size:
                    if wordList[j] in self.sentiwords.keys():
                        print("否定词：{}".format(wordList[i]))
                        W *= -1
                        break
                    j += 1
                continue


            for key in self.degree.keys():    #  判断属于degree中的哪一类程度词
                if signal == 1:
                    break
                if wordList[i] in self.degree[key]:
                    j = i + 1
                    while j < len(wordList) and j - i < window_size:
                        if wordList[j] in self.sentiwords.keys():
                            print("程度词：{}".format(wordList[i]))
                            W *= self.degreeStrength[key]
                            signal = 1
                            break
                        j += 1

            if signal == 0 and wordList[i] in self.sentiwords.keys():  #  判断是否属于情感词典
                print("情感词：{}".format(wordList[i]))
                ind = self.sentiment.index(self.sentiwords[wordList[i]][0])
                sentistrength[ind] += self.sentiwords[wordList[i]][1]*W
                W = 1
        return sentistrength


    def sentimentClassify(self,sentimentList):
        """
        依据情感强度，对句子进行情感分类
        :param sentscore: 类型：array: 每一行对应用户文本每句话的情感强度，每一列对应不同的情感类别
        :return:  类型：array， 每个句子下的情感类别

        """

        Max = max(abs(sentimentList))
        if Max < self.threshold:  #  小于阈值，输出默认类别
            return "默认"
        else:
            category = np.argmax(np.array(abs(sentimentList)))
            if sentimentList[category] < 0:  #  在该类中的强度是反向的，输出对立类别
                temp = self.sentiment[category]
                return self.opposite[temp]

            else:  #  输出该类别
                return self.sentiment[category]


    def addSentiwords(self,base_path,addList):
        with codecs.open(base_path + "/sentiwords/additional.csv","a",encoding = "utf-8") as f:
            writer = csv.writer(f,dialect="excel")
            if addList[0] not in self.additional.keys():
                writer.writerow(addList)
