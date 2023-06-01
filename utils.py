"""
author: Quang Vo
e-mail: quang.vo@sinch.com
last modified: Mar 24, 2023
"""
import gensim
from gensim import corpora, models
from scipy import spatial
import jieba
import numpy as np
from numpy.linalg import norm

class SimilarityFunctions():
    def __init__(self, texts):
        self.texts = texts
        self.lTexts = [jieba.lcut(str(text)) for text in self.texts]
        self.dictionary = corpora.Dictionary(self.lTexts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.lTexts]
        self.tfidf = models.TfidfModel(self.corpus)


    def similarity_word_level(self, groundTruth, keyWord):
        # TF-IDF similarity
        groundTruthBOW = self.dictionary.doc2bow(jieba.lcut(str(groundTruth)))
        groundTruthTFIDF = dict(self.tfidf[groundTruthBOW])

        keyWordBOW = self.dictionary.doc2bow(jieba.lcut(str(keyWord)))
        keyWordTFIDF = dict(self.tfidf[keyWordBOW])

        tfidfMatrix = [[keyWordTFIDF.get(k,0), groundTruthTFIDF.get(k,0)] for k in groundTruthTFIDF.keys()]
        v1 = np.array(tfidfMatrix)[:, 0]
        v2 = np.array(tfidfMatrix)[:, 1]

        tfidfSim = np.dot(v1, v2)/(norm(v1) * norm(v2))


        # IOU
        uniqueWordGroundTruth = set(jieba.lcut(str(groundTruth)))
        uniqueWordKeyWord = set(jieba.lcut(str(keyWord)))

        intersection = set(uniqueWordGroundTruth) & set(uniqueWordKeyWord)
        union = set(uniqueWordGroundTruth) | set(uniqueWordKeyWord)

        iou = len(intersection) / max(len(union),1)


        return iou, tfidfSim

