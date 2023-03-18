

import base64
import logging
import json
import pandas as pd
import sys
import nbformat as nbf
import os
import ast
import timeit
import numpy as np
import networkx as nx
import re

current_dir=os.getcwd()
if os.path.exists(f"{current_dir}/interface/retrieval_engine_module/module2"):
    module2_path=f"{current_dir}/interface/retrieval_engine_module/module2"
    flg_loading=True
    sys.path.append(module2_path)
elif os.path.exists(f"{current_dir}/Similarity_Search_on_Computational_Notebooks/retrieval_system/interface/retrieval_engine_module"):
    module2_path=f"{current_dir}/Similarity_Search_on_Computational_Notebooks/retrieval_system/interface/retrieval_engine_module"
    flg_loading=True
    sys.path.append(module2_path)
else:
    logging.error("module2 is not found in code_relatedness.py.")
    logging.error(f"{current_dir}/interface/retrieval_engine_module/module2")
    exit(1)
from py2neo import Node, Relationship, NodeMatcher, RelationshipMatcher
from module2.config import config
from module2.utils.funclister import FuncLister
from module2.db.table_db import generate_graph, pre_vars
from module2.search.search_prov_code import ProvenanceSearch
#from lib import CodeComparer


class CodeRelatedness:
    def __init__(self):
        pass


    def calc_code_rel_by_jaccard_index(self, code_A, code_B):
        #logging.info("use jaccard similarity to measure code rel")

        splited_A=re.split('\n|\t| |,|=',code_A)
        word_set_A=[]
        for l in splited_A:
            if l=='':
                continue
            l=l.split(".")
            word_set_A.extend(l)
        #print(word_set_A)
        splited_B=re.split('\n|\t| |,|=',code_B)
        word_set_B=[]
        for l in splited_B:
            if l=='':
                continue
            l=l.split(".")
            word_set_B.extend(l)
        #print(word_set_B)
        return self.jaccard_similarity_coefficient(word_set_A, word_set_B)



    # 参考 https://dukesoftware00.blogspot.com/2014/11/java-compute-source-code-similarity.html
    def replace_Multiple_Space_And_Tab_To_Single_Space(self, line):
        line = line.strip().replace("\t", " ")
        while "  " in line:
            line = line.replace("  ", " ")
        return line

    def calc_Hash_For_Each_Line(self, lines):
        hash_set=[]
        for line in lines:
            line = self.replace_Multiple_Space_And_Tab_To_Single_Space(line)
            if line is None or line == "":
                continue
            hash_set.append(self.hash_func(line,0))
        return hash_set

    def hash_func(self, string, seed):
        ret_hash = seed
        for one_char in string:
            ret_hash = ret_hash*101 + ord(one_char)
        return ret_hash


    @staticmethod
    def jaccard_similarity_coefficient(colA:list, colB:list) -> float:
        """
        The Jaccard similarity between two sets A and B is defined as
        |intersection(A, B)| / |union(A, B)|.
        集合Aと集合Bのジャカード類似度を計算する．

        Args:
            colA list[any]: データ値の集合.
            colB list[any]: データ値の集合.

        Returns:
            float: ジャカード類似度．
        """
        try:
            if min(len(colA), len(colB)) == 0:
                return 0
            colA = np.array(colA) #numpyの型に変換?
            # 疑問: colBは変換しなくて良いのか？
            colB = np.array(colB) #numpyの型に変換?
            union = len(np.union1d(colA, colB))
            inter = len(np.intersect1d(colA, colB))
            return float(inter) / float(union)
        except:
            return 0


    def cosine_similarity(self, vecA, vecB):
        if min(len(vecA), len(vecB)) == 0 or len(vecA) != len(vecB):
            return 0
        vecA = np.array(vecA) #numpyの型に変換?
        vecAvecB_sum=0
        A2_sum=0
        B2_sum=0
        for i in range(len(vecA)):
            vecAvecB_sum+=vecA[i]*vecB[i]
            A2_sum=vecA[i]*vecA[i]
            B2_sum=vecB[i]*vecB[i]
        return float(vecAvecB_sum) / (float(A2_sum)*float(B2_sum))


    def calc_code_rel_by_hash_and_jaccard_index(self, code_A, code_B):
        # 参考 https://dukesoftware00.blogspot.com/2014/11/java-compute-source-code-similarity.html
        if type(code_A)==type("string"):
            code_A=code_A.split("\n")
        if type(code_B)==type("string"):
            code_B=code_B.split("\n")
        hash_set_A=self.calc_Hash_For_Each_Line(code_A)
        hash_set_B=self.calc_Hash_For_Each_Line(code_B)
        return self.jaccard_similarity_coefficient(hash_set_A, hash_set_B)
