"""
TODO:calc_v_microbenchmark["load"]を廃止
TODO:コード類似度計算をcode_similarityに統一
"""

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
import timeout_decorator
import matplotlib.pyplot as plt
import copy
import random
from typing import Tuple, Union, List # annotations

from py2neo import Node, Relationship, NodeMatcher, RelationshipMatcher
from mymodule.config import config
from module2.utils.funclister import FuncLister
from module2.db.table_db import generate_graph
from mymodule.code_relatedness import CodeRelatedness
from mymodule.code_similarity import CodeSimilarity
from module2.db.schemamapping import SchemaMapping
#from lib import CodeComparer

NUM_STR_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]



class WorkflowMatching:
    """
    (注意)グラフ(graph_db)は有向グラフではなく，リレーションで定義されている．
    PostgreSQLでアクセスする方のデータベースのスキーマ: (code, cell_id)
    (つまりrowが`code`，`code_id`で，それぞれセルのソースコードとセルIDが保存されている．)
    グラフ(Neo4jで保存)はプロパティが`Cell`と`Var`がある．
    プロパティ`Cell`のノードは，キーに`cell_10`などの`cell_`+セルID(?)で構成される文字列，値にそのセル番号のソースコードをもつ．
    プロパティ`Var`のノードは，キー`name`に対して値は文字列{cell_id}_{var}_{nb_name}の構造をしている．

    code_dict dict{str: int}:
        各セルに対して，code_dictのkeyはセルのソースコード(Base64にエンコードしたもの)，valueはセルID．

    postgres_eng: PostgreSQLのインスタンス
    
    graph_db (Graph): py2neoのクラス`Graph`のインスタンス
    """
    def __init__(self, postgres_eng, graph_eng, sim_col_thres:float=0.5, w_c:float=1, w_v:float=1, w_l:float=1, w_d:float=1, k:int=6, change_next_node_thres:float=0.8, valid_nb_name_file_path:str="../データセット/valid_nb_name.txt", flg_juneau:bool=False):
        """
        PostgreSQLのインスタンスとpy2neoのインスタンスをインスタンスにセットする．

        Args:
            postgres_eng: PostgreSQLのインスタンス
            graph_eng (Graph): py2neoのクラス`Graph`のインスタンス．インスタンス変数graph_dbに格納．
            query (list[Node]): クエリのワークフローグラフ．ラベルは"Cell"がセルノード，"Var"が表形式データのノード，"OneWildcard"は1つのいずれかのノード，"AnyWildcard"は0個以上のいずれかのノード
        """
        self.calc_code_sim_method="jaccard"
        self.graph_db = graph_eng
        self.postgres_eng = postgres_eng
        self.query_workflow = [] #query_nodeの名前の方がいいかも
        self.query_relationship={}
        self.root_list=[]
        self.root_list2=[]
        self.query_cell_code={}
        self.query_root=None
        self.query_table={}
        self.ans_list={}
        self.nb_score={}
        self.valid_nb_name=[] # list[cleaned_nb_name]: 全てのワークフロー化したnbについて表データをDBに保存しているわけではないので一時的に使用
        self.k=k
        self.calculated_sim = {} #dict{tuple(Node, Node): float}: {(ノードq, ノードdb): 類似度}． 複数のワークフローに登場する類似度があるため，一回計算した組み合わせのノードの類似度は取っておく．
        self.calculated_lib_sim = {} #dict{tuple(Node, Node): float}: {(ノードq, ノードdb): 類似度}． 複数のワークフローに登場する類似度があるため，一回計算した組み合わせのノードの類似度は取っておく．
        self.library={} #dict{str: list[str]}: {nb name: [library used in the nb]}
        self.display_type={}
        self.display_type_and_cell_id={}
        self.cell_source_code={} #dict{str: dict{int: str}}: {nb name: {cell id: source code}}
        self.sim_col_thres=sim_col_thres
        self.nb_name_group_id={} # {str: int}. {NB名: グループID}関数set_group_idでセット
        self.w_c=w_c
        self.w_v=w_v
        self.w_l=w_l
        self.w_d=w_d
        self.query_workflow_info={}
        self.db_workflow_info={}
        self.valid_nb_name_file_path=valid_nb_name_file_path
        self.change_next_node_thres=change_next_node_thres
        self.calc_time_sum=0
        self.top_k_score=[]
        self.flg_running_faster={}
        self.class_code_relatedness=CodeRelatedness()
        self.invalid_by_workflow_structure={"invalid":set(), "valid":set()}
        self.calc_v_count=0
        self.nb_node_list={}
        self.init_each_calc_time_sum()

        self.query_acol_set={}
        self.query_scma_dict={}
        self.calc_v_microbenchmark={"load":0, "set":0, "column_rel":0, "sort":0, "table_rel":0}
        
        self.flg_juneau=flg_juneau
        self.dict_nb_name_and_cleaned_nb_name={}
        self.dict_nb_name_and_cleaned_nb_name2={}
        self.load_json={}
        self.flg_use_artifical_dataset=False
        self.schema_element_sample_col={}
        self.data_profile={} # {int: [str]}, {data ID: a list of data node names}
        self.datanametodataID = None #self.readDataCache(cachemode="small") #データIDを取得（同じデータは同じID）
        self.flg_useDataCache = False



    def init_each_calc_time_sum(self):
        """
        Initialize self.each_calc_time_sum.
        
        時間計測のための辞書の初期化．
        """
        self.each_calc_time_sum={"Cell":0.0, "Var":0.0, "Display_data":0.0, "Library":0.0}

    def set_each_w(self, w_c:float=None, w_v:float=None, w_l:float=None, w_d:float=None):
        """Set each weight."""
        if not w_c is None:
            logging.info(f"Set code sim weight = {w_c}")
            self.w_c=w_c
        if not w_v is None:
            logging.info(f"Set data sim weight = {w_v}")
            self.w_v=w_v
        if not w_l is None:
            logging.info(f"Set library sim weight = {w_l}")
            self.w_l=w_l
        if not w_d is None:
            logging.info(f"Set output sim weight = {w_d}")
            self.w_d=w_d

    def set_k(self,k:int):
        """
        Set a natural number k for top-k search on computational notebooks.

        Args:
            k (int): a natural number for top-k search on computational notebooks.
        """
        self.k=k

    def set_query_attr(self):
        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.set_query_workflow_info_Display_data()

    def set_sim_col_thres(self, sim_col_thres:float):
        self.sim_col_thres=sim_col_thres
    
    # 読み込み
    # クエリでは不使用　実装の際の便宜用
    def load_calculated_sim(self, calculated_sim_path:str):
        """
        For development.
        開発用．
        calculated_sim_pathで指定するパスからJSON形式の計算済み類似度を取得しself.calculated_simに格納．
        """
        with open(calculated_sim_path, mode="r") as f:
            load_json=f.read()

        load_json=json.loads(load_json)
        calculated_sim={}
        for list_row in load_json:
            calculated_sim[(list_row[0], list_row[1])]=list_row[2]
        self.calculated_sim=calculated_sim

    # 書き込み
    # クエリでは不使用　実装の際の便宜用
    def store_calculated_sim(self, calculated_sim_path:str):
        """
        For development.
        開発用．
        変数self.calculated_simに格納される計算済み類似度をJSON形式に変換し，
        calculated_sim_pathで指定するパスに格納．
        """
        store_json=[]
        for key, val in self.calculated_sim.items():
            store_json.append([key[0], key[1], val])
        store_json=json.dumps(store_json)
        with open(calculated_sim_path, mode="w") as f:
            f.write(store_json)

    #実際のNBの一部 
    # 実行時間の測定に利用
    def make_query_for_user_exp(self, workflow_type, cleaned_nb_name, cell_id_list, var_name_list, display_type_list, data_size=1):
        """For experiments."""
        if workflow_type==1:
            self.QueryGraph, self.query_root, cell_count, var_count, display_count_dict, max_indegree, max_outdegree=self.make_query_base_graph_1(display_type_list)
            logging.info(f"using function :make_query_base_graph_1")
        elif workflow_type==2:
            self.QueryGraph, self.query_root, cell_count, var_count, display_count_dict, max_indegree, max_outdegree=self.make_query_base_graph_2(display_type_list)
            logging.info(f"using function :make_query_base_graph_2")
        elif workflow_type==3:
            self.QueryGraph, self.query_root, cell_count, var_count, display_count_dict, max_indegree, max_outdegree=self.make_query_base_graph_3(display_type_list)
            logging.info(f"using function :make_query_base_graph_3")
        elif workflow_type==4:
            self.QueryGraph, self.query_root, cell_count, var_count, display_count_dict, max_indegree, max_outdegree=self.make_query_base_graph_cell_only_1()
            logging.info(f"using function :make_query_base_graph_cell_only_1")
            
        if len(cell_id_list)!=cell_count:
            logging.info(f"err: len of given cell_id_list is invalid. valid len:{cell_count}")
        if len(var_name_list)!=var_count:
            logging.info(f"err: len of given cell_id_list is invalid. valid len:{var_count}")

        i=1
        self.query_table={}
        for var_name in var_name_list:
            q_table=self.fetch_var_table(f"{var_name}")
            self.query_table[f"query_var{i}"]=q_table[:int(len(q_table.index)*data_size)]
            i+=1

        code_table=self.fetch_source_code_table(f"{cleaned_nb_name}")
        i=1
        self.query_cell_code={}
        for cid in cell_id_list:
            code=code_table[code_table["cell_id"]==cid]["cell_code"].values
            code="".join(list(code))
            self.query_cell_code[f"cell_query_{i}"]=code
            i+=1

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db(f"{cleaned_nb_name}")
        self.query_workflow_info={"Cell": cell_count, "Var": var_count, "Display_data": display_count_dict, "max_indegree": max_indegree, "max_outdegree": max_outdegree}
        self.set_query_workflow_info_Display_data()


    def make_query_base_graph_1(self, display_type_list):
        """For experiments."""
        base_QueryGraph=nx.DiGraph()
        for i in range(1,5):
            base_QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        base_QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        base_QueryGraph.add_node("query_display1", node_type="Display_data", display_type=display_type_list[0], cell_id="2")
        base_QueryGraph.add_node("query_display2", node_type="Display_data", display_type=display_type_list[1], cell_id="4")

        base_QueryGraph.add_edge("cell_query_1", "cell_query_2")
        base_QueryGraph.add_edge("cell_query_2", "cell_query_3")
        base_QueryGraph.add_edge("cell_query_2", "query_var1")
        base_QueryGraph.add_edge("cell_query_2", "query_display1")
        base_QueryGraph.add_edge("cell_query_3", "cell_query_4")
        base_QueryGraph.add_edge("cell_query_4", "query_display2")
        cell_count=4
        var_count=1
        display_count_dict={}
        for d_type in display_type_list:
            if d_type not in display_count_dict:
                display_count_dict[d_type]=0
            display_count_dict[d_type]+=1
        max_indegree=1
        max_outdegree=3
        query_root="cell_query_1"
        return base_QueryGraph, query_root, cell_count, var_count, display_count_dict, max_indegree, max_outdegree

    def make_query_base_graph_2(self, display_type_list):
        """For experiments."""
        base_QueryGraph=nx.DiGraph()
        for i in range(1,5):
            base_QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        base_QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        base_QueryGraph.add_node("query_var2", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        base_QueryGraph.add_node("query_display1", node_type="Display_data", display_type=display_type_list[0], cell_id="2")

        base_QueryGraph.add_edge("cell_query_1", "cell_query_2")
        base_QueryGraph.add_edge("cell_query_2", "cell_query_3")
        base_QueryGraph.add_edge("cell_query_2", "query_var1")
        base_QueryGraph.add_edge("cell_query_2", "query_display1")
        base_QueryGraph.add_edge("cell_query_3", "cell_query_4")
        base_QueryGraph.add_edge("cell_query_4", "query_var2")
        cell_count=4
        var_count=1
        display_count_dict={}
        for d_type in display_type_list:
            if d_type not in display_count_dict:
                display_count_dict[d_type]=0
            display_count_dict[d_type]+=1
        max_indegree=1
        max_outdegree=3
        query_root="cell_query_1"
        return base_QueryGraph, query_root, cell_count, var_count, display_count_dict, max_indegree, max_outdegree

    def make_query_base_graph_3(self, display_type_list):
        """For experiments."""
        base_QueryGraph=nx.DiGraph()
        for i in range(1,5):
            base_QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        base_QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        base_QueryGraph.add_node("query_var2", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        base_QueryGraph.add_node("query_display1", node_type="Display_data", display_type=display_type_list[0], cell_id="2")

        base_QueryGraph.add_edge("cell_query_1", "cell_query_2")
        base_QueryGraph.add_edge("cell_query_1", "query_var1")
        base_QueryGraph.add_edge("cell_query_1", "query_display1")
        base_QueryGraph.add_edge("cell_query_2", "cell_query_3")
        base_QueryGraph.add_edge("cell_query_3", "cell_query_4")
        base_QueryGraph.add_edge("cell_query_4", "query_var2")
        cell_count=4
        var_count=1
        display_count_dict={}
        for d_type in display_type_list:
            if d_type not in display_count_dict:
                display_count_dict[d_type]=0
            display_count_dict[d_type]+=1
        max_indegree=1
        max_outdegree=3
        query_root="cell_query_1"
        return base_QueryGraph, query_root, cell_count, var_count, display_count_dict, max_indegree, max_outdegree

    def make_sample_query_nx1_2(self):
        """For experiments."""
        self.QueryGraph=nx.DiGraph()

        # 頂点のセット
        for i in range(1,4):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")

        # 辺のセット
        self.QueryGraph.add_edge("cell_query_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_2", "query_var1")

        # 根を設定
        self.query_root="cell_query_1"

        # テーブルのセット
        self.query_table["query_var1"]=self.fetch_var_table("3_df_edaonindiancuisine")

        # ソースコードのセット
        code_table=self.fetch_source_code_table("edaonindiancuisine")
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==7]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code

        # ライブラリのセット
        self.query_lib=self.fetch_all_library_from_db("edaonindiancuisine")

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_workflow_info={"Cell": 3, "Var": 1, "Display_data": {}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    def make_sample_query_nx1_3_old(self):
        """For experiments."""
        self.QueryGraph=nx.DiGraph()

        # 頂点のセット
        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        self.QueryGraph.add_node("query_var2", node_type="Var", data_type="pandas.core.frame.DataFrame")

        # 辺のセット
        self.QueryGraph.add_edge("cell_query_1", "cell_query_2", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "query_var1", edge_type="Contains")
        self.QueryGraph.add_edge("cell_query_4", "query_var2", edge_type="Contains")

        # 根を設定
        self.query_root="cell_query_1"
        
        # テーブルのセット
        self.query_table["query_var1"]=self.fetch_var_table("3_df_edaonindiancuisine")
        self.query_table["query_var2"]=self.fetch_var_table("5_df_edaonindiancuisine")

        # ソースコードのセット
        code_table=self.fetch_source_code_table("edaonindiancuisine")
        code=code_table[code_table["cell_id"]==2]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==4]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_4"]=code

        # ライブラリのセット
        self.query_lib=self.fetch_all_library_from_db("edaonindiancuisine")

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_workflow_info={"Cell": 4, "Var": 2, "Display_data": {}, "max_indegree": 1, "max_outdegree": 2}
    
    #実際のNBの一部 
    def make_sample_query_nx1_3(self):
        """For experiments."""
        self.QueryGraph=nx.DiGraph()

        # 頂点のセット
        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        self.QueryGraph.add_node("query_var2", node_type="Var", data_type="pandas.core.frame.DataFrame")

        # 辺のセット
        self.QueryGraph.add_edge("cell_query_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4")
        self.QueryGraph.add_edge("cell_query_2", "query_var1")
        self.QueryGraph.add_edge("cell_query_4", "query_var2")

        # 根を設定
        self.query_root="cell_query_1"
        
        # テーブルのセット
        self.query_table["query_var1"]=self.fetch_var_table("3_df_edaonindiancuisine")
        self.query_table["query_var2"]=self.fetch_var_table("5_df_edaonindiancuisine")

        # ソースコードのセット
        code_table=self.fetch_source_code_table("edaonindiancuisine")
        code=code_table[code_table["cell_id"]==2]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==4]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_4"]=code

        # ライブラリのセット
        self.query_lib=self.fetch_all_library_from_db("edaonindiancuisine")

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_workflow_info={"Cell": 4, "Var": 2, "Display_data": {}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部 
    def make_sample_query_nx1_4(self):
        """For experiments."""
        self.QueryGraph=nx.DiGraph()

        # 頂点のセット
        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")

        # 辺のセット
        self.QueryGraph.add_edge("cell_query_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4")
        self.QueryGraph.add_edge("cell_query_2", "query_var1")

        # 根を設定
        self.query_root="cell_query_1"
        
        # テーブルのセット
        self.query_table["query_var1"]=self.fetch_var_table("3_df_edaonindiancuisine")

        # ソースコードのセット
        code_table=self.fetch_source_code_table("edaonindiancuisine")
        code=code_table[code_table["cell_id"]==2]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==4]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_4"]=code

        # ライブラリのセット
        self.query_lib=self.fetch_all_library_from_db("edaonindiancuisine")

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_workflow_info={"Cell": 4, "Var": 1, "Display_data": {}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()


    def make_sample_query_nx2_1(self):
        """For experiments."""
        self.QueryGraph=nx.DiGraph()

        for i in range(1,4):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        # 参考: Neo4j保存時は node_display = Node("Display_data", name=node_name, data_type=display_data, cell_id=cell_id, nb_name=nb_name) 
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="png", cell_id=3)

        self.QueryGraph.add_edge("cell_query_1", "cell_query_2", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "query_var1", edge_type="Contains")
        self.QueryGraph.add_edge("cell_query_2", "query_display1", edge_type="Display")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("10_df_housepriceprediction")
        cell_id=1
        code_table=self.fetch_source_code_table("housepriceprediction")
        for i in range(1,4):
            while cell_id not in code_table["cell_id"]:
                cell_id+=1
            code=code_table[code_table["cell_id"]==cell_id]["cell_code"].values
            code="".join(list(code))
            self.query_cell_code[f"cell_query_{i}"]=code
        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("housepriceprediction")
        self.query_workflow_info={"Cell": 3, "Var": 1, "Display_data": {"png": 1}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    def make_sample_query_nx2_2(self):
        """For experiments."""
        self.QueryGraph=nx.DiGraph()

        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="DataFrame", cell_id="4")

        self.QueryGraph.add_edge("cell_query_1", "cell_query_2", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "query_var1", edge_type="Contains")
        self.QueryGraph.add_edge("cell_query_4", "query_display1", edge_type="Display")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("3_df_edaonindiancuisine")

        code_table=self.fetch_source_code_table("edaonindiancuisine")
        code=code_table[code_table["cell_id"]==2]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==4]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_4"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("edaonindiancuisine")
        self.query_workflow_info={"Cell": 4, "Var": 1, "Display_data": {"DataFrame": 1}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    # ユーザ実験に利用
    def make_sample_query_nx2_3(self):
        """For experiments."""
        self.QueryGraph=nx.DiGraph()

        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="DataFrame", cell_id="2")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="DataFrame", cell_id="4")

        self.QueryGraph.add_edge("cell_query_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_2", "query_var1")
        self.QueryGraph.add_edge("cell_query_2", "query_display1")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4")
        self.QueryGraph.add_edge("cell_query_4", "query_display2")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("3_df_edaonindiancuisine")

        code_table=self.fetch_source_code_table("edaonindiancuisine")
        code=code_table[code_table["cell_id"]==2]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==4]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_4"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("edaonindiancuisine")
        self.query_workflow_info={"Cell": 4, "Var": 1, "Display_data": {"DataFrame": 2}, "max_indegree": 1, "max_outdegree": 3}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    # ユーザ実験に利用
    def make_sample_query_nx2_3_with_wildcard(self):
        """For experiments."""
        self.QueryGraph=nx.DiGraph()

        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        for i in range(1,3):
            self.QueryGraph.add_node(f"wildcard_{i}", node_type="AnyWildcard")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="DataFrame", cell_id="2")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="DataFrame", cell_id="4")

        self.QueryGraph.add_edge("cell_query_1", "wildcard_1")
        self.QueryGraph.add_edge("wildcard_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "wildcard_2")
        self.QueryGraph.add_edge("wildcard_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_2", "query_var1")
        self.QueryGraph.add_edge("cell_query_2", "query_display1")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4")
        self.QueryGraph.add_edge("cell_query_4", "query_display2")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("3_df_edaonindiancuisine")

        code_table=self.fetch_source_code_table("edaonindiancuisine")
        code=code_table[code_table["cell_id"]==2]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==4]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_4"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("edaonindiancuisine")
        self.query_workflow_info={"Cell": 4, "Var": 1, "Display_data": {"DataFrame": 2}, "max_indegree": 1, "max_outdegree": 3}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    def make_sample_query_nx2_3_another1(self): #論文用
        """For experiments."""
        # DataSet19, did-you-said-basics-eda-ml-for-very-beginners.ipynb
        self.QueryGraph=nx.DiGraph()

        for i in range(1,4):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="DataFrame")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="text")
        self.QueryGraph.add_node("query_display3", node_type="Display_data", display_type="png")

        self.QueryGraph.add_edge("cell_query_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_1", "query_var1")
        self.QueryGraph.add_edge("cell_query_1", "query_display1")
        self.QueryGraph.add_edge("cell_query_2", "query_display2")
        self.QueryGraph.add_edge("cell_query_3", "query_display3")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("4_Data_sicsedamlforverybeginners")

        code_table=self.fetch_source_code_table("sicsedamlforverybeginners")
        code=code_table[code_table["cell_id"]==4]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==7]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("sicsedamlforverybeginners")
        self.query_workflow_info={"Cell": 3, "Var": 1, "Display_data": {"DataFrame": 1, "text":1, "png":1}, "max_indegree": 1, "max_outdegree": 3}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    # ユーザ実験に利用
    def make_sample_query_nx2_3_another2(self): #論文用
        """For experiments."""
        # DataSet16, mobile-phone-pricing-predictions.ipynb
        self.QueryGraph=nx.DiGraph()

        for i in range(1,4):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="text")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="png")

        self.QueryGraph.add_edge("cell_query_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_1", "query_var1")
        self.QueryGraph.add_edge("cell_query_2", "query_display1")
        self.QueryGraph.add_edge("cell_query_3", "query_display2")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("32_X2_lephonepricingpredictions")

        code_table=self.fetch_source_code_table("lephonepricingpredictions")
        code=code_table[code_table["cell_id"]==32]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==33]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==35]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("lephonepricingpredictions")
        self.query_workflow_info={"Cell": 3, "Var": 1, "Display_data": {"text":1, "png":1}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    def make_sample_query_nx2_3_another2_with_wildcard_1(self): #論文用
        """For experiments."""
        # DataSet16, mobile-phone-pricing-predictions.ipynb
        self.QueryGraph=nx.DiGraph()

        for i in range(1,4):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell")
        for i in range(2,3):
            self.QueryGraph.add_node(f"wildcard_{i}", node_type="AnyWildcard")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="text")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="png")

        #self.QueryGraph.add_edge("cell_query_1", "wildcard_1")
        #self.QueryGraph.add_edge("wildcard_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "wildcard_2")
        self.QueryGraph.add_edge("wildcard_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_1", "query_var1")
        self.QueryGraph.add_edge("cell_query_2", "query_display1")
        self.QueryGraph.add_edge("cell_query_3", "query_display2")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("32_X2_lephonepricingpredictions")

        code_table=self.fetch_source_code_table("lephonepricingpredictions")
        code=code_table[code_table["cell_id"]==32]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==33]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==35]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("lephonepricingpredictions")
        self.query_workflow_info={"Cell": 3, "Var": 1, "Display_data": {"text":1, "png":1}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    def make_sample_query_nx2_3_another2_with_wildcard_2(self): #論文用
        """For experiments."""
        # DataSet16, mobile-phone-pricing-predictions.ipynb
        self.QueryGraph=nx.DiGraph()

        for i in range(1,4):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell")
        for i in range(1,3):
            self.QueryGraph.add_node(f"wildcard_{i}", node_type="AnyWildcard")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="text")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="png")

        self.QueryGraph.add_edge("cell_query_1", "wildcard_1")
        self.QueryGraph.add_edge("wildcard_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "wildcard_2")
        self.QueryGraph.add_edge("wildcard_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_1", "query_var1")
        self.QueryGraph.add_edge("cell_query_2", "query_display1")
        self.QueryGraph.add_edge("cell_query_3", "query_display2")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("32_X2_lephonepricingpredictions")

        code_table=self.fetch_source_code_table("lephonepricingpredictions")
        code=code_table[code_table["cell_id"]==32]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==33]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==35]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("lephonepricingpredictions")
        self.query_workflow_info={"Cell": 3, "Var": 1, "Display_data": {"text":1, "png":1}, "max_indegree": 1, "max_outdegree": 2}
        self.set_query_workflow_info_Display_data()


    #実際のNBの一部
    # ユーザ実験に利用
    def make_sample_query_nx2_3_another3(self): #論文用
        """For experiments."""
        # DataSet19, video-games-industry-made-simple.ipynb
        self.QueryGraph=nx.DiGraph()

        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="DataFrame")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="DataFrame")
        self.QueryGraph.add_node("query_display3", node_type="Display_data", display_type="png")

        self.QueryGraph.add_edge("cell_query_1", "cell_query_2")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4")
        self.QueryGraph.add_edge("cell_query_1", "query_var1")
        self.QueryGraph.add_edge("cell_query_1", "query_display1")
        self.QueryGraph.add_edge("cell_query_2", "query_display2")
        self.QueryGraph.add_edge("cell_query_4", "query_display3")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("1_df_eogamesindustrymadesimple")

        code_table=self.fetch_source_code_table("eogamesindustrymadesimple")
        code=code_table[code_table["cell_id"]==1]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==2]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_4"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("eogamesindustrymadesimple")
        self.query_workflow_info={"Cell": 4, "Var": 1, "Display_data": {"DataFrame": 2, "png":1}, "max_indegree": 1, "max_outdegree": 3}
        self.set_query_workflow_info_Display_data()

    #実際のNBの一部
    # ユーザ実験に利用
    def make_sample_query_nx2_3_another3_with_wildcard(self): #論文用
        """For experiments."""
        # DataSet19, video-games-industry-made-simple.ipynb
        self.QueryGraph=nx.DiGraph()

        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        for i in range(1,3):
            self.QueryGraph.add_node(f"wildcard_{i}", node_type="AnyWildcard")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="DataFrame")
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="DataFrame")
        self.QueryGraph.add_node("query_display3", node_type="Display_data", display_type="png")

        self.QueryGraph.add_edge("cell_query_1", "wildcard_1")
        self.QueryGraph.add_edge("wildcard_1", "cell_query_2")
        #self.QueryGraph.add_edge("cell_query_2", "cell_query_3")
        self.QueryGraph.add_edge("cell_query_2", "wildcard_2")
        self.QueryGraph.add_edge("wildcard_2", "cell_query_3")
        #self.QueryGraph.add_edge("cell_query_3", "wildcard_3")
        #self.QueryGraph.add_edge("wildcard_3", "cell_query_4")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4")
        self.QueryGraph.add_edge("cell_query_3", "cell_query_4")
        self.QueryGraph.add_edge("cell_query_1", "query_var1")
        self.QueryGraph.add_edge("cell_query_1", "query_display1")
        self.QueryGraph.add_edge("cell_query_2", "query_display2")
        self.QueryGraph.add_edge("cell_query_4", "query_display3")

        self.query_root="cell_query_1"
        self.query_table["query_var1"]=self.fetch_var_table("1_df_eogamesindustrymadesimple")

        code_table=self.fetch_source_code_table("eogamesindustrymadesimple")
        code=code_table[code_table["cell_id"]==1]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_1"]=code
        code=code_table[code_table["cell_id"]==2]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_2"]=code
        code=code_table[code_table["cell_id"]==3]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_3"]=code
        code=code_table[code_table["cell_id"]==5]["cell_code"].values
        code="".join(list(code))
        self.query_cell_code[f"cell_query_4"]=code

        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.query_lib=self.fetch_all_library_from_db("eogamesindustrymadesimple")
        self.query_workflow_info={"Cell": 4, "Var": 1, "Display_data": {"DataFrame": 2, "png":1}, "max_indegree": 1, "max_outdegree": 3}
        self.set_query_workflow_info_Display_data()

    #不使用？
    def set_query_workflow_info(self):
        self.query_workflow_info={"Cell": 0, "Var": 0, "Display_data": {"all":0}, "max_indegree": 0, "max_outdegree": 0}
        for n in self.QueryGraph.nodes():
            indegree=0
            for i in self.QueryGraph.predecessors(n):
                indegree+=1
            self.query_workflow_info["max_indegree"]=max(indegree, self.query_workflow_info["max_indegree"])
            
            outdegree=0
            for i in self.QueryGraph.successors(n):
                outdegree+=1
            self.query_workflow_info["max_outdegree"]=max(outdegree, self.query_workflow_info["max_outdegree"])

            node_type=self.attr_of_q_node_type[n]
            if node_type=="Display_data":
                display_type=self.attr_of_q_display_type[n]
                if display_type not in self.query_workflow_info[node_type]:
                    self.query_workflow_info[node_type][display_type]=0
                self.query_workflow_info[node_type][display_type]+=1
                self.query_workflow_info["Display_data"]["all"]+=1
                continue
            if node_type=="AnyWildcard":
                pass
            else:
                self.query_workflow_info[node_type]+=1

    def set_query_workflow_info_Display_data(self):
        sum_i=0
        for i in self.query_workflow_info["Display_data"].values():
            sum_i+=i
        self.query_workflow_info["Display_data"]["all"]=sum_i

    #setDB-cleaned.ipynbで利用しているmymoduleからコピー
    def __parse_code(self, code_list:str):
        test = FuncLister()
        all_code = ""
        line2cid = {}
        

        lid = 1
        fflg = False # defとreturnの行の間だけTrue
        for cid, cell in enumerate(code_list):
            codes = cell.split("\\n")
            new_codes = []
            for code in codes:
                if code[:3].lower() == "def":
                    fflg = True
                    continue

                temp_code = code.strip(" ")
                temp_code = temp_code.strip("\t")

                if temp_code[:6].lower() == "return":
                    fflg = False
                    continue

                code = code.strip("\n")
                code = code.strip(" ")
                code = code.split('"')
                code = "'".join(code)
                code = code.split("\\")
                code = "".join(code)

                line2cid[lid] = cid
                lid = lid + 1
                if len(code) == 0:
                    continue
                if code[0] == "%":
                    continue
                if code[0] == "#":
                    continue

                try:
                    ast.parse(code)
                    if not fflg:
                        new_codes.append(code)
                except:
                    pass
                    #logging.info(code)

            all_code = all_code + "\n".join(new_codes) + "\n"

        all_code = all_code.strip("\n")

        tree = ast.parse(all_code)
        test.visit(tree)
        return test.dependency, line2cid, all_code

    #setDB-cleaned.ipynbからコピー
    #不使用
    def cleaning_one_cell_code(self,cellcode):
        flg1,flg2=0,0
        new_cellcode=[]
        for c in cellcode:
            c2=c.strip()
            if c2=="" or c2[0]=="%":
                continue
                
            prev_flg1 = flg1
            prev_flg2 = flg2
                        
            ind=0
            for each_str in c2:
                if each_str=="\"" or each_str == "\'":
                        flg1= (flg1+1)%2
                if flg1!=1 and (each_str=="(" or each_str=="{" or each_str=="["):
                    flg2+=1
                if flg1!=1 and (each_str==")" or each_str=="}" or each_str=="]"):
                    flg2-=1
                if flg1!=1 and each_str=="#":
                    c=c[:c.index("#")]
                    break
                ind+=1
                    
            c2=c.strip()
            if c2=="":
                continue

            if prev_flg1==0 and prev_flg2==0:
                insert_code=c
            else:
                insert_code+=c
                        
            if flg1==0 and flg2==0:
                new_cellcode.append(insert_code)
            #print(f"cell id {i} : flg {flg1}, {flg2}")
        return new_cellcode


    def gather_nb_name_from_gdb(self):
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
        
        #あとで消す
        nb_name_list=set()
        all_n=matcher.match().all()
        for n in all_n:
            if "nb_name" in n:
                nb_name=n["nb_name"]
            else:
                nb_name=n["name"]
                nb_name=nb_name[nb_name.rfind("_")+1:]
            nb_name_list.add(nb_name)
        return nb_name_list

    # set_db_graphで使用
    def add_node_to_graph(self, node) -> bool:
        """
        Add nodes to DiGraph 'self.G'

        Args:
            node (Node): py2neo instance.
        """
        err=False
        if node.has_label("Var"):
            nb_name=node["name"]
            nb_name=nb_name[nb_name.rfind("_")+1:]
            if nb_name not in self.valid_nb_name:
                return True
            #self.G.add_node(node["name"], node_type="Var", nb_name=nb_name, data_type=node["data_type"])
            self.G.add_node(node["name"], node_type="Var", nb_name=nb_name)
        elif node.has_label("Cell"):
            if node["nb_name"] not in self.valid_nb_name:
                return True
            self.G.add_node(node["name"], node_type="Cell", nb_name=node["nb_name"], real_cell_id=node["real_cell_id"])
        elif node.has_label("Display_data"):
            if node["nb_name"] not in self.valid_nb_name:
                return True
            #self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
            self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"])
        return err
    
    def init_db_workflow_info(self, nb_name:str):
        if nb_name not in self.db_workflow_info:
            self.db_workflow_info[nb_name]={"Cell": 0, "Var": 0, "Display_data": {}, "max_indegree": 0, "max_outdegree": 0}

    # グラフをデータベースから読み込むために使用
    def set_db_graph(self):
        """Neo4Jからグラフを読み込み，self.Gに格納する．"""
        self.set_valid_nb_name()

        self.G = nx.DiGraph()
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化

        node_list = matcher.match().all()
        for start_node in node_list: #全てのノードをnetworkxのDiGraphに追加
            err=self.add_node_to_graph(start_node)
            #err=self.add_node_to_graph_and_set_workflow_info(start_node)
            if err:
                continue
        G_has_node=self.G.has_node
        G_add_edge=self.G.add_edge
        for start_node in node_list:
            if not G_has_node(start_node["name"]):
                continue
            rel_list = r_matcher.match((start_node, ), "Successor").all() + r_matcher.match((start_node, ), "Contains").all() + r_matcher.match((start_node, ), "Usedby").all() + r_matcher.match((start_node, ), "Display").all()
            for rel in rel_list:
                end_node=rel.end_node
                if not G_has_node(end_node["name"]):
                    continue
                #    err=self.add_node_to_graph(end_node)
                #    if err:
                #        continue
                G_add_edge(start_node["name"], end_node["name"])

        self.attr_of_db_node_type=nx.get_node_attributes(self.G, "node_type")
        self.attr_of_db_nb_name=nx.get_node_attributes(self.G, "nb_name")
        self.attr_of_db_real_cell_id=nx.get_node_attributes(self.G, "real_cell_id")
        self.attr_of_db_display_type=nx.get_node_attributes(self.G, "display_type")
        self.set_all_label_node_list()
        self.set_nb_node_list()

    # ベンチマーク用: すでに読み込んだグラフから読み込むために使用
    def set_db_graph2(self, graph):
        """
        キャッシュを読み込み，グラフをself.Gに格納する．
        
        Args:
            graph: self.Gのキャッシュ．
        """
        self.set_valid_nb_name()

        self.G = graph
        self.attr_of_db_node_type=nx.get_node_attributes(self.G, "node_type")
        self.attr_of_db_nb_name=nx.get_node_attributes(self.G, "nb_name")
        self.attr_of_db_real_cell_id=nx.get_node_attributes(self.G, "real_cell_id")
        self.attr_of_db_display_type=nx.get_node_attributes(self.G, "display_type")
        self.set_all_label_node_list()
        self.set_nb_node_list()
    
    def set_nb_node_list(self):
        """self.nb_node_listにCode, Data, Outputのラベルのノード数を，計算ノートブックごとに格納．"""
        for node in self.attr_of_db_nb_name:
            nb_name = self.attr_of_db_nb_name[node]
            node_type=self.attr_of_db_node_type[node]
            if nb_name not in self.nb_node_list:
                self.nb_node_list[nb_name]={"Cell":[], "Var":[], "Display_data":[]}
            self.nb_node_list[nb_name][node_type].append(node)

    """
    def init_table_knn_sim_list(self, table_name):
        with self.psql_engine.connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {config.sql.knn_sim_list}.knn_sim_list " 
                f"(start_node TEXT, end_node TEXT, sim );" # VARCHAR(M)は, 最大M文字数の可変長文字列の型.
            )

    def store_knn_graph_via_sql(self, knn_k=5):
        #sqlでストアする場合
        store_df = pd.DataFrame()
        if not self.all_node_list:
            self.set_all_label_node_list()
        for node_type in self.all_node_list:
            node_list=self.all_node_list[node_type]
            rel_list=[]
            for node_nameA in node_list:
                for node_nameB in node_list:
                    if node_nameA == node_nameB:
                        continue
                    rel_list.append(node_nameB, self.calc_rel_with_timecount(node_nameA, node_nameB))
                sorted_rel_list=sorted(rel_list, key=lambda x:x[1], reverse=True)
                for i in range(knn_k):
                    node_nameB=sorted_rel_list[i][0]
                    sim=sorted_rel_list[i][1]
                    store_df.append([node_nameA, node_nameB, sim])
        if len(store_df.index) > 0:
            store_df.columns=["start_node", "end_node", "sim"]
            try:
                store_df.to_sql(
                        name=f"knn_sim_list",
                        con=conn,
                        schema=config.sql.knn_sim_list, # schema name(not table definitions)
                        if_exists="replace", # 既にあった場合は置き換え
                        index=False,
                    )
            except Exception as e:
                logging.error(f"Unable to store 'knn_sim_list' due to error {e}")
    """                
        
    def set_all_label_node_list(self):
        """self.all_node_listにCode, Data, Outputラベルごとにノードの数を格納．"""
        self.all_node_list={"Cell": [], "Var": [], "Display_data": []}
        node_list = self.attr_of_db_node_type
        for node in node_list:
            node_type=self.attr_of_db_node_type[node]
            self.all_node_list[node_type].append(node)

    def countup_display_data(self, nb_name:str, display_type:str):
        if display_type not in self.db_workflow_info[nb_name]["Display_data"]:
            self.db_workflow_info[nb_name]["Display_data"][display_type]=0
        self.db_workflow_info[nb_name]["Display_data"][display_type]+=1
        
    def get_db_workflow_info(self):

        self.attr_of_db_node_type=nx.get_node_attributes(self.G, "node_type")
        self.attr_of_db_nb_name=nx.get_node_attributes(self.G, "nb_name")
        self.attr_of_db_real_cell_id=nx.get_node_attributes(self.G, "real_cell_id")
        self.attr_of_db_display_type=nx.get_node_attributes(self.G, "display_type")
        self.set_all_label_node_list()
        self.set_nb_node_list()
        
        self.db_workflow_info={}
        node_db_list=self.G.nodes # list[str]
        for node in node_db_list:
            nb_name=self.attr_of_db_nb_name[node]
            node_type=self.attr_of_db_node_type[node]
            indegree=0
            for i in self.G.predecessors(node):
                indegree+=1
            outdegree=0
            for i in self.G.successors(node):
                outdegree+=1
                
            self.init_db_workflow_info(nb_name)

            if node in self.attr_of_db_display_type:
                display_type = self.attr_of_db_display_type[node]
                self.countup_display_data(nb_name, display_type)
            else:
                self.db_workflow_info[nb_name][node_type]+=1

            self.db_workflow_info[nb_name]["max_indegree"]=max(indegree, self.db_workflow_info[nb_name]["max_indegree"])
            self.db_workflow_info[nb_name]["max_outdegree"]=max(outdegree, self.db_workflow_info[nb_name]["max_outdegree"])
            
    def get_sta_of_dataset(self):
        self.get_db_workflow_info()
        self.set_all_label_node_list()
        num_of_nb=len(self.db_workflow_info)
        if num_of_nb < 1:
            return {}, {}, {}
        num_of_node={}
        for node_type in self.all_node_list:
            num_of_node[node_type]={"max":0, "min":1001001, "avg":0, "sum":0}
        num_of_node["Any"]={"max":0, "min":1001001, "avg":len(self.G.nodes)/num_of_nb, "sum":len(self.G.nodes)}
        num_of_node["max_indegree"]={"max":0, "min":1001001, "avg":0, "sum":0}
        num_of_node["max_outdegree"]={"max":0, "min":1001001, "avg":0, "sum":0}
        sta_of_table_data={"max":0, "min":1001001, "avg":0, "sum":0}
        for node_type in self.all_node_list:
            num_of_node[node_type]["sum"]=len(self.all_node_list[node_type])
            num_of_node[node_type]["avg"]=len(self.all_node_list[node_type])/num_of_nb
        sta_of_display_data={} # Display_dataのそれぞれの型ごとにノード数の統計

        for nb_name in self.db_workflow_info:
            num_of_any_type_node=0
            for node_type in self.all_node_list:
                if node_type == "Display_data": # Display_dataの数
                    d_num=0
                    for display_type in self.db_workflow_info[nb_name][node_type]:
                        n_num=self.db_workflow_info[nb_name][node_type][display_type]
                        d_num+=n_num
                        if display_type not in sta_of_display_data:
                            sta_of_display_data[display_type]={"max":0, "min":1001001, "avg":0, "sum":0}
                        sta_of_display_data[display_type]["min"]=min(n_num, sta_of_display_data[display_type]["min"])
                        sta_of_display_data[display_type]["max"]=max(n_num, sta_of_display_data[display_type]["max"])
                        sta_of_display_data[display_type]["sum"]=n_num+sta_of_display_data[display_type]["sum"]
                    num_of_node[node_type]["min"]=min(d_num, num_of_node[node_type]["min"])
                    num_of_node[node_type]["max"]=max(d_num, num_of_node[node_type]["max"])
                    num_of_any_type_node+=d_num
                else:
                    n_num=self.db_workflow_info[nb_name][node_type]
                    num_of_node[node_type]["min"]=min(n_num, num_of_node[node_type]["min"])
                    num_of_node[node_type]["max"]=max(n_num, num_of_node[node_type]["max"])
                    num_of_any_type_node+=n_num
            num_of_node["Any"]["min"]=min(num_of_any_type_node, num_of_node["Any"]["min"])
            num_of_node["Any"]["max"]=max(num_of_any_type_node, num_of_node["Any"]["max"])
            num_of_node["max_indegree"]["min"]=min(num_of_node["max_indegree"]["min"], self.db_workflow_info[nb_name]["max_indegree"])
            num_of_node["max_indegree"]["max"]=max(num_of_node["max_indegree"]["max"], self.db_workflow_info[nb_name]["max_indegree"])
            num_of_node["max_indegree"]["sum"]+=self.db_workflow_info[nb_name]["max_indegree"]
            num_of_node["max_outdegree"]["min"]=min(num_of_node["max_outdegree"]["min"], self.db_workflow_info[nb_name]["max_outdegree"])
            num_of_node["max_outdegree"]["max"]=max(num_of_node["max_outdegree"]["max"], self.db_workflow_info[nb_name]["max_outdegree"])
            num_of_node["max_outdegree"]["sum"]+=self.db_workflow_info[nb_name]["max_outdegree"]
        sta_of_display_data[display_type]["avg"]=sta_of_display_data[display_type]["sum"]/num_of_nb

        log_count=0
        for var_name in self.all_node_list["Var"]:
            table_size = self.get_table_size3(self.fetch_var_table(var_name))
            sta_of_table_data["min"] = min(table_size, sta_of_table_data["min"])
            sta_of_table_data["max"] = max(table_size, sta_of_table_data["max"])
            sta_of_table_data["sum"] += table_size
            log_count+=1
            if log_count%50==0:
                logging.info(f"collecting tables' data... : {log_count} done")
        sta_of_table_data["avg"]=sta_of_table_data["sum"]/num_of_nb
        return num_of_node, sta_of_table_data, sta_of_display_data

    def get_sta_of_dataset_node(self):
        self.get_db_workflow_info()
        self.set_all_label_node_list()
        num_of_nb=len(self.db_workflow_info)
        if num_of_nb < 1:
            return {}, {}, {}
        num_of_node={}
        for node_type in self.all_node_list:
            num_of_node[node_type]={"max":0, "min":1001001, "avg":0, "sum":0}
        num_of_node["Any"]={"max":0, "min":1001001, "avg":len(self.G.nodes)/num_of_nb, "sum":len(self.G.nodes)}
        num_of_node["max_indegree"]={"max":0, "min":1001001, "avg":0, "sum":0}
        num_of_node["max_outdegree"]={"max":0, "min":1001001, "avg":0, "sum":0}
        for node_type in self.all_node_list:
            num_of_node[node_type]["sum"]=len(self.all_node_list[node_type])
            num_of_node[node_type]["avg"]=len(self.all_node_list[node_type])/num_of_nb
        sta_of_display_data={} # Display_dataのそれぞれの型ごとにノード数の統計

        for nb_name in self.db_workflow_info:
            num_of_any_type_node=0
            for node_type in self.all_node_list:
                if node_type == "Display_data": # Display_dataの数
                    d_num=0
                    for display_type in self.db_workflow_info[nb_name][node_type]:
                        n_num=self.db_workflow_info[nb_name][node_type][display_type]
                        d_num+=n_num
                        if display_type not in sta_of_display_data:
                            sta_of_display_data[display_type]={"max":0, "min":1001001, "avg":0, "sum":0}
                        sta_of_display_data[display_type]["min"]=min(n_num, sta_of_display_data[display_type]["min"])
                        sta_of_display_data[display_type]["max"]=max(n_num, sta_of_display_data[display_type]["max"])
                        sta_of_display_data[display_type]["sum"]=n_num+sta_of_display_data[display_type]["sum"]
                    num_of_node[node_type]["min"]=min(d_num, num_of_node[node_type]["min"])
                    num_of_node[node_type]["max"]=max(d_num, num_of_node[node_type]["max"])
                    num_of_any_type_node+=d_num
                else:
                    n_num=self.db_workflow_info[nb_name][node_type]
                    num_of_node[node_type]["min"]=min(n_num, num_of_node[node_type]["min"])
                    num_of_node[node_type]["max"]=max(n_num, num_of_node[node_type]["max"])
                    num_of_any_type_node+=n_num
            num_of_node["Any"]["min"]=min(num_of_any_type_node, num_of_node["Any"]["min"])
            num_of_node["Any"]["max"]=max(num_of_any_type_node, num_of_node["Any"]["max"])
            num_of_node["max_indegree"]["min"]=min(num_of_node["max_indegree"]["min"], self.db_workflow_info[nb_name]["max_indegree"])
            num_of_node["max_indegree"]["max"]=max(num_of_node["max_indegree"]["max"], self.db_workflow_info[nb_name]["max_indegree"])
            num_of_node["max_indegree"]["sum"]+=self.db_workflow_info[nb_name]["max_indegree"]
            num_of_node["max_outdegree"]["min"]=min(num_of_node["max_outdegree"]["min"], self.db_workflow_info[nb_name]["max_outdegree"])
            num_of_node["max_outdegree"]["max"]=max(num_of_node["max_outdegree"]["max"], self.db_workflow_info[nb_name]["max_outdegree"])
            num_of_node["max_outdegree"]["sum"]+=self.db_workflow_info[nb_name]["max_outdegree"]
        
        num_of_node["max_indegree"]["avg"]=num_of_node["max_indegree"]["sum"]/num_of_nb
        num_of_node["max_outdegree"]["avg"]=num_of_node["max_outdegree"]["sum"]/num_of_nb
        for display_type in self.db_workflow_info[nb_name][node_type]:
            sta_of_display_data[display_type]["avg"]=sta_of_display_data[display_type]["sum"]/num_of_nb

        return num_of_node, sta_of_display_data

    def get_sta_of_dataset_table(self):
        self.get_db_workflow_info()
        self.set_all_label_node_list()
        num_of_nb=len(self.db_workflow_info)
        if num_of_nb < 1:
            return {}, {}, {}
        sta_of_table_data={"max":0, "min":1001001, "avg":0, "sum":0, "zero_size":0}
        log_count=0
        num_of_var_node=len(self.all_node_list["Var"])
        logging.info(f"collecting all tables' data starts : {num_of_var_node}")
        for var_name in self.all_node_list["Var"]:
            table_size = self.get_table_size3(self.fetch_var_table(var_name))
            if table_size == 0:
                logging.info(f"size of table '{var_name}' is 0.")
                sta_of_table_data["zero_size"] +=1
                continue
            sta_of_table_data["min"] = min(table_size, sta_of_table_data["min"])
            sta_of_table_data["max"] = max(table_size, sta_of_table_data["max"])
            sta_of_table_data["sum"] += table_size
            log_count+=1
            if log_count%50==0:
                logging.info(f"collecting tables' data... : {log_count}/{num_of_var_node} done")
        logging.info(f"collecting all tables' data is done : {log_count}")
        sta_of_table_data["avg"]=sta_of_table_data["sum"]/num_of_nb
        return sta_of_table_data

    def get_sta_of_dataset_source_code(self):
        sta_of_code_data={}
        sta_of_code_data["all"]={"max":0, "min":1001001, "avg":0, "avg_cell":0,"sum":0}
        cell_num=0
        #cell_num2=0
        for nb_name in self.valid_nb_name:
            sta_of_code_data[nb_name]={"max":0, "min":1001001, "avg":0, "sum":0}
            db_code_table=self.fetch_source_code_table(nb_name)
            db_code_list=db_code_table["cell_code"].values
            #cell_num2+=len(db_code_list)
            for cell_code in db_code_list:
                num_of_lines=0
                for l in cell_code:
                    if l=="":
                        continue
                    num_of_lines+=1
                if num_of_lines==0:
                    continue
                cell_num+=1
                sta_of_code_data[nb_name]["min"]=min(num_of_lines, sta_of_code_data[nb_name]["min"])
                sta_of_code_data[nb_name]["max"]=max(num_of_lines, sta_of_code_data[nb_name]["max"])
                sta_of_code_data[nb_name]["sum"]+=num_of_lines
            if len(db_code_list)!=0:
                sta_of_code_data[nb_name]["avg"]=sta_of_code_data[nb_name]["sum"]/len(db_code_list)
            else:
                sta_of_code_data[nb_name]["avg"]= 0
        #logging.info(f"cell_num2: {cell_num2}")
        for nb_name in sta_of_code_data:
            sta_of_code_data["all"]["min"]=min(sta_of_code_data[nb_name]["min"], sta_of_code_data["all"]["min"])
            sta_of_code_data["all"]["max"]=max(sta_of_code_data[nb_name]["max"], sta_of_code_data["all"]["max"])
            sta_of_code_data["all"]["sum"]+=sta_of_code_data[nb_name]["sum"]
        if len(db_code_list)!=0:
            sta_of_code_data["all"]["avg"]=sta_of_code_data["all"]["sum"]/len(db_code_list)
        else:
            sta_of_code_data["all"]["avg"]= 0
        if cell_num!=0:
            sta_of_code_data["all"]["avg_cell"]=sta_of_code_data["all"]["sum"]/cell_num
        else:
            sta_of_code_data["all"]["avg_cell"]= 0
        return sta_of_code_data

    def show_sta_of_dataset(self):
        num_of_node, sta_of_display_data=self.get_sta_of_dataset_node()
        sta_of_table_data=self.get_sta_of_dataset_table()
        df_num_of_node = pd.DataFrame()

        for node_type in num_of_node:
            df_num_of_node=df_num_of_node.append([["Node", node_type]+list(num_of_node[node_type].values())])
        df_num_of_node=df_num_of_node.append([["Table", "table data"]+list(sta_of_table_data.values())])
        for display_type in sta_of_display_data:
            df_num_of_node=df_num_of_node.append([["Outputs", display_type]+list(sta_of_display_data[display_type].values())])
            
        return df_num_of_node

    # データベースがうまく作れているノートブック名をセットするのに使用
    def set_valid_nb_name(self):
        """
        self.valid_nb_name_file_pathのパスからファイルを読み込み，
        検索対象とする計算ノートブック名のリストをself.valid_nb_nameに格納する．
        """
        with open(self.valid_nb_name_file_path, mode="r") as f:
            lines=f.readlines()
            for l in lines:
                l=l.strip()
                self.valid_nb_name.append(l) #cleaned_nb_name
        self.valid_nb_name=list(set(self.valid_nb_name))

    # 素朴な手法で使用
    def subgraph_matching(self, limit=None, nb_limit=None, tflg=False):
        if tflg:
            self.ans_list={}
            self.detected_count=0
            if self.flg_running_faster["chk_invalid_by_workflow_structure"]:
                return self.real_subgraph_matching_part_A(limit, nb_limit)
            else:
                return self.real_subgraph_matching_part_B(limit, nb_limit)
        else:
            self.real_subgraph_matching_part_without_timecount(limit, nb_limit)

    # 使用予定：wildcardをクエリに含む
    def subgraph_matching_with_wildcard_in_query(self, limit:int=None, nb_limit:int=None, tflg:bool=False):
        self.ans_list={}
        self.detected_count=0
        return self.real_subgraph_matching_part_A_with_wildcard_in_query(limit, nb_limit)

    
    # 提案手法の交互実行で使用
    def subgraph_matching_weaving(self, limit=None, nb_limit=None, tflg=False, flg_knn=False):
        if tflg:
            self.ans_list={}
            self.detected_count=0
            return self.real_subgraph_matching_part_weaving_with_timecount(limit, nb_limit, flg_knn=flg_knn)
        else:
            self.real_subgraph_matching_part_weaving_without_timecount(limit, nb_limit, flg_knn)
  
    # 使用
    def chk_invalid_by_workflow_structure(self, nb_name):
        """
        return True if num of any workflow components is less than query
        invalid -> True
        valid -> False
        """
        if nb_name in self.invalid_by_workflow_structure["invalid"]:
            return True
        elif nb_name in self.invalid_by_workflow_structure["valid"]:
            return False

        for ele in self.query_workflow_info:
            if ele not in self.db_workflow_info[nb_name]:
                #logging.info(f"'{ele}' not in self.db_workflow_info['{nb_name}']")
                #return True
                continue
            if ele == "Display_data":
                num_of_d_sum=0
                for num_of_d in self.db_workflow_info[nb_name][ele].values():
                    num_of_d_sum+=num_of_d
                if self.query_workflow_info[ele]["all"] > num_of_d_sum:
                    self.invalid_by_workflow_structure["invalid"].add(nb_name)
                    return True
                continue
                #for display_type in self.query_workflow_info[ele]:
                #    if display_type not in self.db_workflow_info[nb_name][ele]:
                #        #logging.info(f"'{display_type}' not in self.db_workflow_info['{nb_name}']['{ele}']")
                #        self.invalid_by_workflow_structure["invalid"].add(nb_name)
                #        return True
                #    elif self.query_workflow_info[ele][display_type] > self.db_workflow_info[nb_name][ele][display_type]:
                #        #logging.info(f"self.query_workflow_info['{ele}']['{display_type}'] > self.db_workflow_info['{nb_name}']['{ele}']['{display_type}']")
                #        self.invalid_by_workflow_structure["invalid"].add(nb_name)
                #        return True
            elif self.query_workflow_info[ele] > self.db_workflow_info[nb_name][ele]:
                #logging.info(f"self.query_workflow_info['{ele}'] > self.db_workflow_info['{nb_name}']['{ele}']")
                self.invalid_by_workflow_structure["invalid"].add(nb_name)
                return True
        self.invalid_by_workflow_structure["valid"].add(nb_name)
        return False
        
    # 使用
    # 工夫「ワークフロー構造の考慮」あり
    def real_subgraph_matching_part_A(self, limit=None, nb_limit=None):
        """
        工夫「ワークフロー構造の考慮」あり
        先に関数set_db_graphでデータベースから読み込んでグラフをself.Gに格納している必要あり．
        サブグラフマッチングを一括で行う．
        クエリグラフはself.QueryGraph

        少しだけ参考: https://github.com/qinshimeng18/subgraph-matching/blob/master/main.py

        self.attributes:
            self.ans_list (list[list[Node]]): サブグラフマッチングで検索したサブグラフのノード集合のリストのリスト．

        Returns:
            float: この関数全体の処理時間
            float: サブグラフが検出された開始ノードに対し，次の開始ノードのセットなどを除く処理時間
            int-like: サブグラフが検出された開始ノードの数
            float: サブグラフが検出されなかった開始ノードに対し，次の開始ノードのセットなどを除く処理時間
            int-like: サブグラフが検出されなかった開始ノードの数
            int-like: 検出されたサブグラフの個数
        """
        start_time1 = timeit.default_timer()
        time2=0 # マッチするサブグラフがあったときの時間の合計
        n_count2=0
        time3=0 # マッチするサブグラフがなかったときの時間の合計
        n_count3=0
        self.detected_count=0
        
        visited=[]
        matched=[]

        #有効なnb名をセット
        self.set_valid_nb_name()

        query_label=self.attr_of_q_node_type[self.query_root]
        node_db_list=[]
        if query_label in ["Cell", "Var", "Display_data"]:
            node_db_list=self.all_node_list[query_label]
        else: # クエリのrootがOneWildcardまたはAnyWildcardのとき
            node_db_list=self.G.nodes # list[str]


        for node_db in node_db_list:
            # node_db (str): データベースのノートブックをワークフロー化したグラフのノード
            nb_name=self.attr_of_db_nb_name[node_db]
            if nb_name not in self.valid_nb_name:
                continue
            if self.chk_invalid_by_workflow_structure(nb_name):
                continue

            ret_list=[]
            roop_count=0
            matched=[]
            visited=[]
            if nb_name not in self.ans_list:
                self.ans_list[nb_name]=[]
            self.flg_detection=False
            
            start_time2 = timeit.default_timer()
            #detected_subgraph=self.rec_new4(ret_list, self.query_root, node_db, visited, matched, nb_name)
            _=self.rec_new4_2(ret_list, self.query_root, node_db, visited, matched, nb_name)
            end_time2 = timeit.default_timer()
                
            if self.flg_detection:
                n_count2+=1
                time2+=(end_time2-start_time2)
            else:
                n_count3+=1
                time3+=(end_time2-start_time2)
        end_time1 = timeit.default_timer()
        time1 = end_time1 - start_time1

        logging.info("subgraph matching completed!")
        nb_count=len(self.ans_list)
        logging.info(f"time1: {time1}")
        if n_count2==0:
            logging.info(f"exist subgraph root: {time2}, {n_count2}")
        else:
            logging.info(f"exist subgraph root: {time2}, {n_count2}, per one root: {time2/n_count2}")
        if n_count3==0:
            logging.info(f"no subgraph root: {time3}, {n_count3}")
        else:
            logging.info(f"no subgraph root: {time3}, {n_count3}, per one root: {time3/n_count3}")
        logging.info(f"detected subgraph: {self.detected_count}")
        logging.info(f"calculated num of nb: {nb_count}")
        if nb_count != 0:
            logging.info(f"time par nb: {time1/nb_count}")
            logging.info(f"num of detected subgraph par nb: {self.detected_count/nb_count}")
        return time1,time2,n_count2,time3,n_count3,self.detected_count, nb_count

    # 使用
    # 工夫「ワークフロー構造の考慮」なし
    def real_subgraph_matching_part_B(self, limit=None, nb_limit=None):
        """
        工夫「ワークフロー構造の考慮」なし
        先に関数set_db_graphでデータベースから読み込んでグラフをself.Gに格納している必要あり．
        サブグラフマッチングを一括で行う．
        クエリグラフはself.QueryGraph

        少しだけ参考: https://github.com/qinshimeng18/subgraph-matching/blob/master/main.py

        self.attributes:
            self.ans_list (list[list[Node]]): サブグラフマッチングで検索したサブグラフのノード集合のリストのリスト．

        Returns:
            float: この関数全体の処理時間
            float: サブグラフが検出された開始ノードに対し，次の開始ノードのセットなどを除く処理時間
            int-like: サブグラフが検出された開始ノードの数
            float: サブグラフが検出されなかった開始ノードに対し，次の開始ノードのセットなどを除く処理時間
            int-like: サブグラフが検出されなかった開始ノードの数
            int-like: 検出されたサブグラフの個数
        """
        start_time1 = timeit.default_timer()
        time2=0
        n_count2=0
        time3=0
        n_count3=0
        self.detected_count=0
        
        visited=[]
        matched=[]

        #有効なnb名をセット
        self.set_valid_nb_name()

        query_label=self.attr_of_q_node_type[self.query_root]
        node_db_list=[]
        if query_label in ["Cell", "Var", "Display_data"]:
            node_db_list=self.all_node_list[query_label]
        else: # クエリのrootがOneWildcardまたはAnyWildcardのとき
            node_db_list=self.G.nodes # list[str]


        for node_db in node_db_list:
            # node_db (str): データベースのノートブックをワークフロー化したグラフのノード
            nb_name=self.attr_of_db_nb_name[node_db]
            if nb_name not in self.valid_nb_name:
                continue

            ret_list=[]
            roop_count=0
            matched=[]
            visited=[]
            if nb_name not in self.ans_list:
                self.ans_list[nb_name]=[]
            self.flg_detection=False
            
            start_time2 = timeit.default_timer()
            #detected_subgraph=self.rec_new4(ret_list, self.query_root, node_db, visited, matched, nb_name)
            _=self.rec_new4_2(ret_list, self.query_root, node_db, visited, matched, nb_name)
            end_time2 = timeit.default_timer()
                
            if self.flg_detection:
                n_count2+=1
                time2+=(end_time2-start_time2)
            else:
                n_count3+=1
                time3+=(end_time2-start_time2)
        end_time1 = timeit.default_timer()
        time1 = end_time1 - start_time1

        logging.info("subgraph matching completed!")
        nb_count=len(self.ans_list)
        logging.info(f"time1: {time1}")
        if n_count2==0:
            logging.info(f"exist subgraph root: {time2}, {n_count2}")
        else:
            logging.info(f"exist subgraph root: {time2}, {n_count2}, per one root: {time2/n_count2}")
        if n_count3==0:
            logging.info(f"no subgraph root: {time3}, {n_count3}")
        else:
            logging.info(f"no subgraph root: {time3}, {n_count3}, per one root: {time3/n_count3}")
        logging.info(f"detected subgraph: {self.detected_count}")
        logging.info(f"calculated num of nb: {nb_count}")
        if nb_count != 0:
            logging.info(f"time par nb: {time1/nb_count}")
            logging.info(f"num of detected subgraph par nb: {self.detected_count/nb_count}")
        return time1,time2,n_count2,time3,n_count3,self.detected_count, nb_count


    # 使用
    # 工夫「ワークフロー構造の考慮」あり
    def real_subgraph_matching_part_A_with_wildcard_in_query(self, limit=None, nb_limit=None):
        """
        wildcardを含むクエリに対応．
        """
        start_time1 = timeit.default_timer()
        time2=0
        n_count2=0
        time3=0
        n_count3=0
        self.detected_count=0
        
        visited=[]
        matched=[]
        #nb_list=set()

        #有効なnb名をセット
        self.set_valid_nb_name()

        query_label=self.attr_of_q_node_type[self.query_root]
        node_db_list=[]
        #if query_label in ["Cell", "Var", "Display_data"]:
        node_db_list=self.all_node_list[query_label]
        #else: # クエリのrootがOneWildcardまたはAnyWildcardのとき
        #    node_db_list=self.G.nodes # list[str]

        pre_nb_name=None
        for node_db in node_db_list:
            # node_db (str): データベースのノートブックをワークフロー化したグラフのノード
            nb_name=self.attr_of_db_nb_name[node_db]
            # ---------------
            #おそいやつ
            #if nb_name in ["heartdisease", "ithmsandfeatureimportance", "lephonepricingpredictions", "mobilepriceclassification"]:
            #if nb_name in ["heartdisease"]:
            #    continue
            #はやいやつ
            #if nb_name in ["salesprediction", "lysisandvisualizationwine", "winedatasetdatacleaning", "chooseheartovermind", "edanetflix", "acomparisonoffewmlmodels", "decisiontreeclassifier", "mobilepriceprediction", "ricepredictionwith95score", "classification929accuracy"]:
            #    continue
            # --------------
            if nb_name not in self.valid_nb_name:
                continue
            if self.chk_invalid_by_workflow_structure(nb_name):
                continue
            #if nb_name not in nb_list:
            #    nb_list.add(nb_name)
            #if nb_name != pre_nb_name:
            #    logging.info(f"nb_name: {nb_name}")
            #if nb_name in ["heartdisease", "ithmsandfeatureimportance", "lephonepricingpredictions", "mobilepriceclassification"]:
            #    logging.info(f"nb_name: {nb_name}, node_db: {node_db}")

            ret_list=[]
            roop_count=0
            matched=[]
            visited=[]
            if nb_name not in self.ans_list:
                self.ans_list[nb_name]=[]
            self.flg_detection=False
            
            start_time2 = timeit.default_timer()
            #detected_subgraph=self.rec_new4(ret_list, self.query_root, node_db, visited, matched, nb_name)
            _=self.rec_with_wildcard_in_query(ret_list, self.query_root, node_db, visited, matched, nb_name)
            end_time2 = timeit.default_timer()
                
            if self.flg_detection:
                n_count2+=1
                time2+=(end_time2-start_time2)
                #break
            else:
                n_count3+=1
                time3+=(end_time2-start_time2)
            pre_nb_name=nb_name
        end_time1 = timeit.default_timer()
        time1 = end_time1 - start_time1

        logging.info("subgraph matching completed!")
        nb_count=len(self.ans_list)
        logging.info(f"time1: {time1}")
        if n_count2==0:
            logging.info(f"exist subgraph root: {time2}, {n_count2}")
        else:
            logging.info(f"exist subgraph root: {time2}, {n_count2}, per one root: {time2/n_count2}")
        if n_count3==0:
            logging.info(f"no subgraph root: {time3}, {n_count3}")
        else:
            logging.info(f"no subgraph root: {time3}, {n_count3}, per one root: {time3/n_count3}")
        logging.info(f"detected subgraph: {self.detected_count}")
        logging.info(f"calculated num of nb: {nb_count}")
        if nb_count != 0:
            logging.info(f"time par nb: {time1/nb_count}")
            logging.info(f"num of detected subgraph par nb: {self.detected_count/nb_count}")
        return time1,time2,n_count2,time3,n_count3,self.detected_count, nb_count


    def set_flg_subgraph_matching_mode(self, subgraph_matching_mode):
        self.subgraph_matching_mode=subgraph_matching_mode
        logging.info(f"self.subgraph_matching_mode={subgraph_matching_mode}")
    

    # 使用
    def real_subgraph_matching_part_weaving_without_timecount(self, limit=None, nb_limit=None):
        """
        先に関数set_db_graphでデータベースから読み込んでグラフをself.Gに格納している必要あり．
        サブグラフマッチングを一括で行う．
        クエリグラフはself.QueryGraph

        少しだけ参考: https://github.com/qinshimeng18/subgraph-matching/blob/master/main.py

        self.attributes:
            self.ans_list (list[list[Node]]): サブグラフマッチングで検索したサブグラフのノード集合のリストのリスト．

        Returns:
            float: この関数全体の処理時間
            float: サブグラフが検出された開始ノードに対し，次の開始ノードのセットなどを除く処理時間
            int-like: サブグラフが検出された開始ノードの数
            float: サブグラフが検出されなかった開始ノードに対し，次の開始ノードのセットなどを除く処理時間
            int-like: サブグラフが検出されなかった開始ノードの数
            int-like: 検出されたサブグラフの個数
        """
        self.detected_count=0
        
        visited=[]
        matched=[]
        count=0 #あとで消す
        nb_name_list=[] #あとで消す

        #有効なnb名をセット
        self.set_valid_nb_name()

        query_label=self.attr_of_q_node_type[self.query_root]
        node_db_list=[]
        if query_label in ["Cell", "Var"]:
            for node_name, node_label in self.attr_of_db_node_type.items():
                if node_label == query_label:
                    node_db_list.append(node_name)
        else: # クエリのrootがOneWildcardまたはAnyWildcardのとき
            node_db_list=self.G.nodes # list[str]


        for node_db in node_db_list:
            # node_db (str): データベースのノートブックをワークフロー化したグラフのノード
            nb_name=self.attr_of_db_nb_name[node_db]
            if nb_name not in self.valid_nb_name:
                continue
            if self.chk_invalid_by_workflow_structure(nb_name):
                continue
            # *********以下 あとで消す***************************************************
            count+=1 #あとで消す
            if (not limit is None) and count>=limit: #あとで消す
                break #あとで消す
            if  (not nb_limit is None) and len(self.ans_list) >= nb_limit: #あとで消す
                break #あとで消す
            if nb_name not in nb_name_list: #あとで消す
                nb_name_list.append(nb_name) #あとで消す
                logging.info(f"nb name: {nb_name}") #あとで消す
            # *********以上 あとで消す***************************************************

            
            ret_list=[]
            #logging.info("detecting subgraph...")
            roop_count=0
            matched=[]
            visited=[]
            #if self.query_root.has_label("Cell") or self.query_root.has_label("Var"):
                #matched.append((self.query_root["name"], node_db["name"]))
            if nb_name not in self.ans_list:
                self.ans_list[nb_name]=[]
            if nb_name not in self.nb_score:
                self.nb_score[nb_name]=0.0


            current_score=0
            # self.calculated_sim に計算結果を保存してあとから参照できるので，ここでの計算はあとで活用できるためそこまで時間のロスにはならないはず
            if self.flg_prune_under_sim([self.query_root], self.calc_rel(self.query_root, node_db)):
                continue

            self.flg_detection=False
            detected_subgraph=self.rec_new4_weaving(ret_list, self.query_root, node_db, visited, matched, nb_name, current_score)
                

    def count_root_from_db(self):
        with open(self.valid_nb_name_file_path, mode="r") as f:
            lines=f.readlines()
            for l in lines:
                l=l.strip()
                self.valid_nb_name.append(l) #cleaned_nb_name
        self.valid_nb_name=list(set(self.valid_nb_name))
        
        root_count=0
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
        n_list = matcher.match("Cell").all()
        for n in n_list:
            nb_name=n["nb_name"]
            if nb_name not in self.valid_nb_name:
                continue
            r=r_matcher.match((n,),"Parent").first()
            if r is None:
                root_count+=1
        print(f"root count is {root_count}")


    def collect_info_of_workflow_from_db(self):
        """
        Neo4Jからワークフローの統計情報を取得する．
        py2neoのNodeMatcherやRelationshipMacherを利用している．
        """
        node_count={}
        with open(self.valid_nb_name_file_path, mode="r") as f:
            lines=f.readlines()
            for l in lines:
                l=l.strip()
                self.valid_nb_name.append(l) #cleaned_nb_name
        self.valid_nb_name=list(set(self.valid_nb_name))
        
        root_count=0
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
        n_list = matcher.match("Cell").all()
        for n in n_list:
            nb_name=n["nb_name"]
            if nb_name not in self.valid_nb_name:
                continue
            r=r_matcher.match((n,),"Parent").first()
            if r is None:
                node_count[nb_name]={}
                node_count[nb_name]["cell"]=0
                node_count[nb_name]["tablevar"]=0
                node_count[nb_name]["outputs"]=0
                root_count+=1
                stack=[n]
                #node_count[nb_name]["cell"]+=1
                while len(stack)!=0:
                    current_node=stack.pop()
                    if current_node.has_label("Cell"):
                        node_count[nb_name]["cell"]+=1
                    elif current_node.has_label("Var"):
                        node_count[nb_name]["tablevar"]+=1
                    elif current_node.has_label("Display_data"):
                        node_count[nb_name]["outputs"]+=1
                    else:
                        continue
                    r_list=r_matcher.match((current_node,),"Successor").all()+r_matcher.match((current_node,),"Contains").all()+r_matcher.match((current_node,),"Display").all()
                    if r_list is None:
                        continue
                    for r in r_list:
                        stack.append(r.end_node)

        print(f"root count is {root_count}")
        c_count_sum=0
        v_count_sum=0
        o_count_sum=0
        min_c_count=1001001
        min_v_count=1001001
        min_o_count=1001001
        max_c_count=0
        max_v_count=0
        max_o_count=0
        for nb_name in node_count:
            print("nb_name: ",nb_name)
            print(f"cell count is ", node_count[nb_name]["cell"])
            c_count_sum+=node_count[nb_name]["cell"]
            min_c_count=min(min_c_count,node_count[nb_name]["cell"])
            max_c_count=max(max_c_count, node_count[nb_name]["cell"])
            print(f"table var count is ", node_count[nb_name]["tablevar"])
            v_count_sum+=node_count[nb_name]["tablevar"]
            min_v_count=min(min_v_count,node_count[nb_name]["tablevar"])
            max_v_count=max(max_v_count, node_count[nb_name]["tablevar"])
            print(f"display data count is ", node_count[nb_name]["outputs"])
            o_count_sum+=node_count[nb_name]["outputs"]
            min_o_count=min(min_o_count,node_count[nb_name]["outputs"])
            max_o_count=max(max_o_count, node_count[nb_name]["outputs"])
        print("平均cell数: ",c_count_sum/len(node_count))
        print("平均tablevar数: ",v_count_sum/len(node_count))
        print("平均outputs数: ",o_count_sum/len(node_count))
        return c_count_sum, v_count_sum, min_c_count, min_v_count, max_c_count, max_v_count, len(node_count), o_count_sum, min_o_count, max_o_count


    def set_flg_running_faster(self, flg_chk_invalid_by_workflow_structure=True, flg_flg_prune_under_sim=True, flg_optimize_calc_order=True, flg_caching=True, flg_cache_query_table=False):
        self.flg_running_faster["chk_invalid_by_workflow_structure"]=flg_chk_invalid_by_workflow_structure
        self.flg_running_faster["flg_prune_under_sim"]=flg_flg_prune_under_sim
        self.flg_running_faster["flg_optimize_calc_order"]=flg_optimize_calc_order
        self.flg_running_faster["flg_caching"]=flg_caching
        self.flg_running_faster["flg_calc_data_sim_approximately"]=False
        self.flg_running_faster["flg_cache_query_table"]=flg_cache_query_table
        

    # *** メインで使用(一括でサブグラフマッチング) ***
    def rec_new4_2(self, ret_list, snode_q, snode_db, visited, matched, nb_name):
        """
        self.Gを利用して一括でサブグラフマッチング．(先にデータベースから一括でロード)
        クエリグラフはself.QueryGraph
        wildcardがある場合および閉路を持つクエリでは誤ったサブグラフが検出される可能性あり．

        Args:
            ret_list (list)
            snode_q (Node(networkx))
            snode_db (Node(networkx))
            visited (list[str]): visited nodes list of query workflow
            matched (list[tuple(str, str)]): list of matched nodes. tuple is a node of query workflow and a node of notebook workflow from db.
            nb_name (str): notebook name of the node

        Returns:
            list[list[str]]
            list[str]
        """
        #ret_list: 現在探しているマッピングに対して，現在の深さのノード以外もマッピングしているノード間の関係を全て入れたもの

        # 初期化
        next_q_list=[]
        next_db_list=[]

        
        if (self.attr_of_q_node_type[snode_q] not in ["OneWildcard", "AnyWildcard"]) and self.attr_of_db_node_type[snode_db] != self.attr_of_q_node_type[snode_q]:
            logging.info(f"error1: not match node type of workflow. {self.attr_of_q_node_type[snode_q]}, {self.attr_of_db_node_type[snode_db]}")
        #new_matched=list(set(matched+[(snode_q,snode_db)]))
        new_matched=matched+[(snode_q,snode_db)]
        new_visited=list(set(visited+[(snode_q)]))

        # クエリの全てのノードに訪れた場合，終了
        if len(new_visited) == len(self.QueryGraph.nodes): # クエリの全てのノードに訪れたかどうかの判定
            new_matched=list(set(new_matched))
            new_matched.sort()
            if new_matched not in ret_list:
                ret_list.append(new_matched)
                self.flg_detection=True
                self.ans_list[nb_name].append(new_matched)
                self.detected_count+=1
            else:
                pass
                # ここの条件になる場合は存在しないはず -> 修正に伴って検出されるようになった．
                #logging.info("*************detected duplicate.*****************")
                #logging.info(new_matched)
            return ret_list, new_matched, new_visited


        # データNBのノードsnode_dbの子ノードのリストを得る
        next_db_list=list(self.G.successors(snode_db))


        # クエリのノードsnode_qの子ノードを得る
        #if snode_q.has_label("AnyWildcard"):
        n_q=None
        if self.attr_of_q_node_type[snode_q]=="AnyWildcard":
            next_q_list.append(snode_q)
            n_q=snode_q
        else:
            # クエリNBのノードsnode_qの子ノードのリストを得る
            next_q_list.extend(list(self.QueryGraph.successors(snode_q)))
            if len(next_db_list) < len(next_q_list):
                return ret_list, matched, visited
            for n in list(self.QueryGraph.successors(snode_q)):
                if n in new_visited:
                    continue
                n_q=n
                break
        if n_q is None:
            return ret_list, new_matched, new_visited


        flg_not_found=True
        for n_db in next_db_list:
            if (n_q, n_db) in new_matched:
                continue

            if self.attr_of_q_node_type[snode_q]=="AnyWildcard":
                pass
            elif self.attr_of_q_node_type[n_q]=="OneWildcard":
                pass
            elif self.attr_of_q_node_type[n_q]==self.attr_of_db_node_type[n_db]:
                pass
            else:
                continue


            #remaining_node_list=[]
            flg_exist_any_n_q=False
            for n in next_q_list:
                if n is n_q:
                    continue
                if self.attr_of_q_node_type[n]=="AnyWildcard":
                    continue
                if n not in new_visited:
                    #remaining_node_list.append(n)
                    flg_exist_any_n_q=True
            if flg_exist_any_n_q:
            #if len(new_visited) != len(self.QueryGraph.nodes):
                new_new_visited=list(set(new_visited+[n_q]))
                if (self.attr_of_q_node_type[n_q] not in ["OneWildcard", "AnyWildcard"]) and self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    logging.info(f"error2: not match node type of workflow. {self.attr_of_q_node_type[n_q]}, {self.attr_of_db_node_type[n_db]}")
                #new_new_matched=list(set(new_matched+[(n_q, n_db)]))
                new_new_matched=new_matched+[(n_q, n_db)]
                flg_not_found=False
                ret_list, matched2, visited2=self.rec_new4_2(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name) 
                ret_list, matched2, visited2=self.rec_new4_2(ret_list, n_q, n_db, visited2, matched2, nb_name) 
            else:
                flg_not_found=False
                ret_list, matched2, visited2 =self.rec_new4_2(ret_list, n_q, n_db, new_visited, new_matched, nb_name)
        if flg_not_found:
            return ret_list, matched, visited
        return ret_list, matched2, visited2
    
    # *** 完成版　メインで使用　ワイルドカード不問 ***
    def rec_with_wildcard_in_query(self, ret_list, snode_q, snode_db, visited, matched, nb_name):
        """
        wildcardを含むクエリに対応．
        閉路を持つクエリに対応．

        Args:
            parent_node_is_wildcard (bool): True if parent node in query graph is wildcard.

        current_node_is_wildcard (bool): True if current node in query graph is wildcard.
        """
        #ret_list: 現在探しているマッピングに対して，現在の深さのノード以外もマッピングしているノード間の関係を全て入れたもの
        #print(snode_q,snode_db)

        # 初期化
        next_q_list=[]
        next_db_list=[]
        #current_node_is_wildcard=False

        
        if (self.attr_of_q_node_type[snode_q] not in ["OneWildcard", "AnyWildcard"]) and self.attr_of_db_node_type[snode_db] != self.attr_of_q_node_type[snode_q]:
            logging.info(f"error1: not match node type of workflow. {self.attr_of_q_node_type[snode_q]}, {self.attr_of_db_node_type[snode_db]}")
        #new_matched=list(set(matched+[(snode_q,snode_db)]))
        new_visited=list(set(visited+[(snode_q)]))
        if self.attr_of_q_node_type[snode_q]=="AnyWildcard":
            new_matched=list(set(matched))
            #new_visited=list(set(visited))
        else:
            new_matched=matched+[(snode_q,snode_db)]

        # クエリの全てのノードに訪れた場合，終了
        #remaining_list=[]
        flg_exist_any_n_q=False
        for n in self.QueryGraph.nodes():
            #if self.attr_of_q_node_type[n]=="AnyWildcard":
            #    continue
            if n not in new_visited:
                #remaining_list.append(n)
                #print(f"remaining: {n}")
                flg_exist_any_n_q=True
                #break
        #print(f"remaining list: {remaining_list}")
        #if len(new_visited) == len(self.QueryGraph.nodes): # クエリの全てのノードに訪れたかどうかの判定
        if not flg_exist_any_n_q:
            new_matched=list(set(new_matched))
            new_matched.sort()
            edge_failed_flg=False
            for m_tuple_1 in new_matched:
                for m_tuple_2 in new_matched:
                    if self.QueryGraph.has_edge(m_tuple_1[0], m_tuple_2[0]) and not self.G.has_edge(m_tuple_1[1], m_tuple_2[1]):
                        edge_failed_flg=True
                        break
            if new_matched not in ret_list and not edge_failed_flg:
                ret_list.append(new_matched)
                self.flg_detection=True
                self.ans_list[nb_name].append(new_matched)
                self.detected_count+=1
            else:
                pass
                # ここの条件になる場合は存在しないはず -> 修正に伴って検出されるようになった．
                #logging.info("*************detected duplicate.*****************")
                #logging.info(new_matched)
            return ret_list, new_matched, new_visited


        # データNBのノードsnode_dbの子ノードのリストを得る
        if self.attr_of_q_node_type[snode_q]=="AnyWildcard":
            next_db_list_tmp=[]
            stack=[snode_db]
            while(len(stack)!=0):
                n=stack.pop()
                if n in next_db_list_tmp:
                    continue
                next_db_list_tmp.append(n)
                stack.extend(list(self.G.successors(n)))
            next_db_list_tmp=list(set(next_db_list_tmp))
            next_db_list=[]
            for n in next_db_list_tmp:
                next_db_list.append(n)
                #if self.attr_of_db_node_type[n]=="Cell":
                #    next_db_list.append(n)
        else:
            next_db_list=list(self.G.successors(snode_db))


        # クエリのノードsnode_qの子ノードを得る
        n_q=None
        # クエリNBのノードsnode_qの子ノードのリストを得る
        next_q_list=list(self.QueryGraph.successors(snode_q))
        #if len(next_db_list) < len(next_q_list):
        #    return ret_list, matched, visited
        for n in next_q_list:
            if n in new_visited:
                continue
            if self.attr_of_q_node_type[n]!="AnyWildcard":
                continue
            n_q=n
            break
        for n in next_q_list:
            if n in new_visited:
                continue
            if self.attr_of_q_node_type[n]=="AnyWildcard":
                continue
            n_q=n
            break
        if n_q is None:
            return ret_list, new_matched, new_visited


        flg_not_found=True
        for n_db in next_db_list:
            if (n_q, n_db) in new_matched:
                continue

            if self.attr_of_q_node_type[n_q]=="AnyWildcard":
                pass
            elif self.attr_of_q_node_type[n_q]==self.attr_of_db_node_type[n_db]:
                pass
            else:
                continue


            #remaining_node_list=[]
            flg_exist_any_n_q=False
            for n in next_q_list:
                if n is n_q:
                    continue
                #if self.attr_of_q_node_type[n]=="AnyWildcard":
                #    continue
                if n not in new_visited:
                    #remaining_node_list.append(n)
                    flg_exist_any_n_q=True
                    break
            if flg_exist_any_n_q:
            #if len(new_visited) != len(self.QueryGraph.nodes):
                if (self.attr_of_q_node_type[n_q] in ["OneWildcard", "AnyWildcard"]):
                    flg_not_found=False
                    new_new_visited=list(set(new_visited+[n_q]))
                    ret_list, matched2, visited2=self.rec_with_wildcard_in_query(ret_list, n_q, n_db, new_new_visited, new_matched, nb_name) 
                    #ret_list, matched2, visited2=self.rec_with_wildcard_in_query(ret_list, n_q, n_db, visited2, matched2, nb_name) 
                else:
                    if self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                        logging.info(f"error2: not match node type of workflow. {self.attr_of_q_node_type[n_q]}, {self.attr_of_db_node_type[n_db]}")
                    new_new_visited=list(set(new_visited+[n_q]))
                    #new_new_matched=list(set(new_matched+[(n_q, n_db)]))
                    new_new_matched=new_matched+[(n_q, n_db)]
                    flg_not_found=False
                    ret_list, matched2, visited2=self.rec_with_wildcard_in_query(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name) 
                    ret_list, matched2, visited2=self.rec_with_wildcard_in_query(ret_list, n_q, n_db, visited2, matched2, nb_name) 
            else:
                flg_not_found=False
                ret_list, matched2, visited2 =self.rec_with_wildcard_in_query(ret_list, n_q, n_db, new_visited, new_matched, nb_name)
        if flg_not_found:
            return ret_list, matched, visited
        return ret_list, matched2, visited2
    
    # ****** 以下、スコアリングとtop-k検索について ******

    def calc_one_score(self, matched_list):
        current_score=0
        for n_tuple in matched_list:
            n1_name=n_tuple[0]
            n2_name=n_tuple[1]
            current_score+=self.calc_rel_with_timecount(n1_name, n2_name)
        return current_score


    def calc_rel_with_timecount(self, n1_name:str, n2_name:str) -> float:
        """
        Calculate the similarity between n1_name and n2_name and get how long the calculation needed.
        """
        calc_start_time = timeit.default_timer()
        score=self.calc_rel(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        return score

    def calc_rel_c_o_with_timecount(self, n1_name, n2_name):
        calc_start_time = timeit.default_timer()
        score=self.calc_rel_c_o(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        return score

    def calc_rel_o_with_timecount(self, n1_name:str, n2_name:str) -> float:
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel_o(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        return ret

    def calc_rel_v_with_timecount(self, n1_name:str, n2_name:str) -> float:
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel_v(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        return ret

    def calc_rel_c_with_remain_number_with_timecount(self, n1_name:str, n2_name:str, remain_c_count:int) -> Tuple[float, int]:
        """
        Args:
            n1_name (str): A node name of the query graph.
            n2_name (str): A node name of a workflow graph.
            remain_c_count (int): The number of uncalculated code similarity. This number is used for calculating MaxSim in other functions.

        Returns:
            Tuple[float, int]: A tuple of a product of code similarity and the code weight, and the number of uncalculated code similarity.
        """
        calc_start_time = timeit.default_timer()
        score, remain_c_count=self.calc_rel_c_with_remain_number(n1_name, n2_name, remain_c_count)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        return score, remain_c_count

    def calc_rel_v_with_remain_count_with_timecount(self, n1_name, n2_name, remain_v_count):
        calc_start_time = timeit.default_timer()
        ret1, ret2=self.calc_rel_v_with_remain_count(n1_name, n2_name, remain_v_count)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        return ret1, ret2

    """
    def calc_rel_v_approximately_with_timecount(self, n1_name, n2_name):
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel_v_approximately(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        return ret
    """

    def is_code_match(self, nodenameQ, nodenameN):
        if self.attr_of_db_node_type[nodenameN] == "Cell" and self.attr_of_q_node_type[nodenameQ] == "Cell":
            return True
        else:
            return False

    def is_data_match(self, nodenameQ, nodenameN):
        if self.attr_of_db_node_type[nodenameN] == "Var" and self.attr_of_q_node_type[nodenameQ] == "Var":
            return True
        else:
            return False

    def is_output_match(self, nodenameQ, nodenameN):
        if self.attr_of_db_node_type[nodenameN] == "Display_data" and self.attr_of_q_node_type[nodenameQ] == "Display_data":
            return True
        else:
            return False

    def wrapper_calc_code_sim(self, nodenameQ, nodenameN) -> float:
        """
        コード類似度の計算のインタフェース．

        The interface for calculations of code similarity.
        """
        if self.w_c==0:
            return 0
        sim = self.loadContentsSim(nodenameQ, nodenameN)
        if sim is not None:
            return sim
        #n1_real_cell_id=self.attr_of_q_real_cell_id[nodenameQ]
        n2_real_cell_id=self.attr_of_db_real_cell_id[nodenameN]
        #sim = self.calc_cell_rel(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id, ver="jaccard_similarity_coefficient")
        codeQ=self.query_cell_code[nodenameQ]
        cleaned_nb_name=self.attr_of_db_nb_name[nodenameN]
        code_table_B=self.fetch_source_code_table(cleaned_nb_name)
        codeN=code_table_B[code_table_B["cell_id"]==n2_real_cell_id]["cell_code"].values
        codeN="".join(list(codeN))

        if self.calc_code_sim_method=="jaccard":
            sim=self.class_code_relatedness.calc_code_rel_by_jaccard_index(codeQ, codeN)
        elif self.calc_code_sim_method=="levenshtein":
            sim=CodeSimilarity.code_sim_based_on_levenshtein_dist(codeQ, codeN)
        else:
            sim=self.class_code_relatedness.calc_code_rel_by_jaccard_index(codeQ, codeN)

        self.cacheContentsSim(nodenameQ, nodenameN, sim)
        return sim

    def wrapper_calc_data_sim(self, nodenameQ, nodenameN) -> float:
        """
        データ類似度の計算のインタフェース．self.flg_juneauをTrueにしておくとJuneauを使用する．

        The interface for calculations of data similarity.
        """
        if self.w_v==0:
            return 0.0
        if nodenameQ not in self.query_table:
            return 0.0
        sim = self.loadContentsSim(nodenameQ, nodenameN)
        if sim is not None:
            return sim

        if self.flg_juneau:
            # Juneauをデータ類似度計算に使うとき．
            return self.wrapper_calc_one_data_similarity_using_juneau(tnameQ=nodenameQ,tnameN=nodenameN)
        else:
            score = self.inner_calc_data_sim(query_node_name=nodenameQ, tnameN=nodenameN)
            self.cacheContentsSim(nodenameQ, nodenameN, score)
            return score

    def wrapper_calc_output_sim(self, nodenameQ, nodenameN) -> int:
        """
        出力類似度の計算のインタフェース．

        The interface for calculations of output similarity.
        """
        if self.w_d==0:
            return 0
        #if (nodenameQ, nodenameN) in self.calculated_sim:
        #    return self.calculated_sim[(nodenameQ, nodenameN)]
        if nodenameQ not in self.attr_of_q_display_type:
            return 0
        if nodenameN not in self.attr_of_db_display_type:
            return 0
        if self.attr_of_db_display_type[nodenameN] == self.attr_of_q_display_type[nodenameQ]:
            return 1
        else:
            return 0

    def calc_rel(self, n1_name:str, n2_name:str) -> float:
        """
        Return the product of the similarity and the weight.
        n1_nameとn2_nameの類似度と重みの積を返す．
        
        Args:
            n1_name (str): A node name of the query graph.
            n2_name (str): A node name of a workflow graph.
        
        Returns:
            float: 類似度と重みの積．

        Example:
            If w_D=2, the similarity between n1_name node and n2_name node is 0.6, it returns 1.2.

        """
        if self.is_code_match(nodenameQ=n1_name, nodenameN=n2_name):
            if self.w_c > 0:
                return  self.w_c * self.wrapper_calc_code_sim(nodenameQ=n1_name, nodenameN=n2_name)
            else:
                return  self.w_c * (1-self.wrapper_calc_code_sim(nodenameQ=n1_name, nodenameN=n2_name)) * -1
        elif self.is_data_match(nodenameQ=n1_name, nodenameN=n2_name):
            self.calc_v_count+=1
            """
            if self.flg_running_faster["flg_cache_query_table"]:
                sim = self.calc_table_rel_between_query_and_db(n1_name, tableB)
            else:
                tableA=self.query_table[n1_name]
                #if tableA is None:
                #    return 0.0
                sim = self.calc_table_rel(tableA, tableB)
            """
            if self.w_v > 0:
                return self.w_v * self.wrapper_calc_data_sim(nodenameQ=n1_name, nodenameN=n2_name)
            else:
                return self.w_v * (1-self.wrapper_calc_data_sim(nodenameQ=n1_name, nodenameN=n2_name)) * -1
        elif self.is_output_match(nodenameQ=n1_name, nodenameN=n2_name):
            if self.w_d > 0:
                return self.w_d * self.wrapper_calc_output_sim(nodenameQ=n1_name, nodenameN=n2_name)
            else:
                return self.w_d * (1-self.wrapper_calc_output_sim(nodenameQ=n1_name, nodenameN=n2_name)) * -1
        else:
            #logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[n1_name]}, {self.attr_of_db_node_type[n2_name]}")
            return 0.0

    def calc_rel_c_o(self, n1_name:str, n2_name:str) -> float:
        """Call this function instead of 'calc_rel'(=l_theta, in our paper) when we want to calculate small cost similairty."""
        if self.is_code_match(nodenameQ=n1_name, nodenameN=n2_name):
            if self.w_c > 0:
                return  self.w_c * self.wrapper_calc_code_sim(nodenameQ=n1_name, nodenameN=n2_name)
            else:
                return  self.w_c * (1-self.wrapper_calc_code_sim(nodenameQ=n1_name, nodenameN=n2_name)) * -1
        elif self.is_output_match(nodenameQ=n1_name, nodenameN=n2_name):
            if self.w_d > 0:
                return self.w_d * self.wrapper_calc_output_sim(nodenameQ=n1_name, nodenameN=n2_name)
            else:
                return self.w_d * (1-self.wrapper_calc_output_sim(nodenameQ=n1_name, nodenameN=n2_name)) * -1
        #elif self.is_data_match(nodenameQ=n1_name, nodenameN=n2_name):
        #    return 0
        else:
            #logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[n1_name]}, {self.attr_of_db_node_type[n2_name]}")
            return 0.0

    def calc_rel_c_with_remain_number(self, n1_name:str, n2_name:str, remain_c_count:int):
        """
        Call this function instead of 'calc_rel' when you want calculate only code similarity.

        Args:
            n1_name (str): A node name of the query graph.
            n2_name (str): A node name of a workflow graph.
            remain_c_count (int): The number of uncalculated code similarity. This number is used for calculating MaxSim in other functions.
        """
        if self.is_code_match(nodenameQ=n1_name, nodenameN=n2_name):
            remain_c_count-=1
            sim = self.wrapper_calc_code_sim(nodenameQ=n1_name, nodenameN=n2_name)
            if self.w_c > 0:
                weightedsim = self.w_c * sim
            else:
                weightedsim = self.w_c * (1-sim) * -1
            return weightedsim, remain_c_count
        else:
            return 0.0, remain_c_count

    def calc_rel_o(self, n1_name:str, n2_name:str) -> float:
        """Call this function instead of 'calc_rel' when you want calculate only output similarity."""
        if self.is_output_match(nodenameQ=n1_name, nodenameN=n2_name):
            if self.w_d > 0:
                return self.w_d * self.wrapper_calc_output_sim(nodenameQ=n1_name, nodenameN=n2_name)
            else:
                return self.w_d * (1-self.wrapper_calc_output_sim(nodenameQ=n1_name, nodenameN=n2_name)) * -1
        else:
            return 0.0

    def calc_rel_v(self, n1_name:str, n2_name:str) -> float:
        """Call this function instead of 'calc_rel' when you want calculate only data similarity."""
        if self.is_data_match(nodenameQ=n1_name, nodenameN=n2_name):
            self.calc_v_count+=1
            if self.w_v > 0:
                return self.w_v * self.wrapper_calc_data_sim(nodenameQ=n1_name, nodenameN=n2_name)
            else:
                return self.w_v * (1-self.wrapper_calc_data_sim(nodenameQ=n1_name, nodenameN=n2_name)) * -1
        else:
            return 0.0

    def calc_rel_v_with_remain_count(self, n1_name:str, n2_name:str, remain_v_count) -> Tuple[float, int]:
        """
        For calculating data similarity.
        Call this function instead of 'calc_rel'.

        Args:
            n1_name (str): A node name of the query graph.
            n2_name (str): A node name of a workflow graph.
            remain_v_count (int): The number of uncalculated data similarity. This number is used for calculating MaxSim in other functions.

        Returns:
            Tuple[float, int]: A tuple of a product of data similarity and the data weight, and the number of uncalculated data similarity.
        """
        if self.is_data_match(nodenameQ=n1_name, nodenameN=n2_name):
            self.calc_v_count+=1
            remain_v_count-=1
            sim=self.w_v * self.wrapper_calc_data_sim(nodenameQ=n1_name, nodenameN=n2_name)
            if self.w_v > 0:
                weightedsim = self.w_v * sim
            else:
                weightedsim = self.w_v * (1-sim) * -1
            return weightedsim, remain_v_count
        else:
            return 0.0, remain_v_count

    """
    def calc_rel_v_approximately(self, n1_name, n2_name):
        if self.w_v==0:
            return 0
        if self.is_data_match(nodenameQ=n1_name, nodenameN=n2_name):
            self.calc_v_count+=1
            #if (n1_name, n2_name) in self.calculated_sim:
            #    return self.calculated_sim[(n1_name, n2_name)] * self.w_v
            if n1_name not in self.query_table:
                return 0
            tableA=self.query_table[n1_name]
            tableB=self.fetch_var_table(n2_name)
            if tableB is None:
                return 0
            sim = self.calc_table_rel_approximately(tableA, tableB)
            #self.calculated_sim[(n1_name, n2_name)]=sim
            return sim * self.w_v
        else:
            return 0
    """

    # 使用
    def calc_nb_score3(self, w_c:float=None, w_v:float=None, w_l:float=None, w_d:float=None) -> Tuple[int, int]:
        """
        スコア計算部分の時間計測あり
        self.ans_listの集合 --- ノード名(str)

        Args:
            w_c: the weight of code similarity.
            w_v: the weight of data similarity.
            w_l: the weight of library similarity.
            w_d: the weight of output similarity.
        """

        self.init_each_calc_time_sum()
        self.set_each_w(w_c, w_v, w_l, w_d)

        self.nb_score={}
        count=0
        nb_count=0
        lib_list1=self.query_lib
        for nb_name in self.ans_list:
            lib_list2 = self.fetch_all_library_from_db(nb_name)
            lib_score = self.calc_lib_score(lib_list1, lib_list2)
            self.nb_score[nb_name]=0

            for subgraph in self.ans_list[nb_name]:
                current_score=0
                current_score += lib_score * self.w_l

                for n_tuple in subgraph:
                    n1_name=n_tuple[0]
                    n2_name=n_tuple[1]
                    current_score+=self.calc_rel_with_timecount(n1_name, n2_name)

                #logging.info("score is ", current_score)
                if self.nb_score[nb_name] < current_score:
                    self.nb_score[nb_name]=current_score
                count+=1
                #print(self.nb_score)
            nb_count+=1
        return count, nb_count

    # 使用
    def calc_nb_score3_2(self, w_c:float=None, w_v:float=None, w_l:float=None, w_d:float=None, flg_juneau:bool=False, k_juneau:int=20) -> Tuple[int, int]:
        """
        For naive method and micro-benchmark for the proposal method.
        Time each calculations.（スコア計算部分の時間計測あり．）
        self.ans_listの集合 --- ノード名(str)

        Args:
            w_c: the weight of code similarity.
            w_v: the weight of data similarity.
            w_l: the weight of library similarity.
            w_d: the weight of output similarity.
        """
        self.set_flg_running_faster(False, False, False, False)
        self.init_each_calc_time_sum()
        self.set_each_w(w_c, w_v, w_l, w_d)
        self.init_arranged_all_weights()
        if flg_juneau:
            self.set_table_group2() # Juneauを適用するのに必要
            self.calc_all_tables_rel_using_juneau(k_juneau=k_juneau) #追加
            self.init_query_col()
            self.set_all_querytable_col()

        self.nb_score={}
        count=0
        nb_count=0
        lib_list1=self.query_lib
        for nb_name in self.ans_list:
            lib_list2 = self.fetch_all_library_from_db(nb_name)
            lib_score = self.calc_lib_score(lib_list1, lib_list2)
            self.nb_score[nb_name]=0

            for subgraph in self.ans_list[nb_name]:
                g_score=0
                g_score += lib_score * self.w_l

                for n_tuple in subgraph:
                    n1_name=n_tuple[0]
                    n2_name=n_tuple[1]
                    if flg_juneau and self.is_data_match(nodenameQ=n1_name, nodenameN=n2_name):
                        sim = self.loadContentsSim(n1_name, n2_name)
                        if sim is not None: #juneauで計算した類似度以外のデータ類似度はゼロにする
                            g_score+= self.w_v * sim
                        else:
                            pass # g_score+=0
                    else:
                        g_score+=self.calc_rel_with_timecount(n1_name, n2_name)

                #logging.info("score is ", g_score)
                if self.nb_score[nb_name] < g_score:
                    self.nb_score[nb_name]=g_score
                count+=1
                #print(self.nb_score)
            nb_count+=1
        return count, nb_count

    
    # 使用
    def calc_nb_score_with_opt_order(self, w_c=None, w_v=None, w_l=None, w_d=None):
        """
        類似度下限値の利用のみ（計算順の最適化はなし）
        スコア計算部分の時間計測あり
        self.ans_listの集合 --- ノード名(str)

        Args:
            w_c: セル類似度をNBスコアにする際の重み
            w_v: テーブル類似度をNBスコアにする際の重み
        """
        for w in [w_c, w_v, w_l, w_d]:
            if not w is None:
                self.set_each_w(w_c, w_v, w_l, w_d)
                break

        self.nb_score={}
        count=0
        nb_count=0
        lib_list1=self.query_lib
        v_count=len(self.query_table) # 下限値の導出に利用. 1サブグラフあたりのラベルがデータのペアの数.
        for nb_name in self.ans_list:
            lib_list2 = self.fetch_all_library_from_db(nb_name)
            lib_score = self.calc_lib_score(lib_list1, lib_list2)

            current_score=lib_score * self.w_l


            if len(self.nb_score)<self.k:
                k_score=0
            else:
                self.top_k_score=sorted(self.nb_score.items(), key=lambda d: d[1], reverse=True)
                k_score=self.top_k_score[self.k-1][1]

            self.nb_score[nb_name]=0

            for subgraph in self.ans_list[nb_name]:
                g_score=current_score

                for n_tuple in subgraph:
                    #if self.flg_prune_under_sim(remain_v_count, g_score, max(k_score, self.nb_score[nb_name])):
                    #    break
                    n1_name=n_tuple[0]
                    n2_name=n_tuple[1]
                    g_score+=self.calc_rel_with_timecount(n1_name, n2_name)

                if self.nb_score[nb_name] < g_score:
                    self.nb_score[nb_name]=g_score
                count+=1

            nb_count+=1
        return count, nb_count

    # 使用 現状最速
    def calc_nb_score_for_new_proposal_method_2_3(self, w_c:float=None, w_v:float=None, w_l:float=None, w_d:float=None):
        """
        The similairty calculation component of the proposed method.
        スコア計算部分の時間計測あり
        self.ans_listの集合 --- ノード名(str)
        データ関連度の重みがゼロの時はもっと高速な別の関数に切り替える．
        この関数ではデータ関連度を1つ計算するたびに類似度下限値との比較を行う．
        類似度下限値との比較の関数もver2_2より一部改良している．(計算回数を減らす)
        + flg_running_fasterの比較回数を最小限にした．
        コード+出力+ライブラリ→データの順に計算順序を最適化．
        """
        self.init_query_col()
        self.set_all_querytable_col()
        self.init_each_calc_time_sum()
        self.nb_score={}

        for w in [w_c, w_v, w_l, w_d]:
            if not w is None:
                self.set_each_w(w_c, w_v, w_l, w_d)

        self.init_arranged_all_weights()

        if not self.flg_running_faster["flg_prune_under_sim"]:
            # For micro-benchmark.
            return self.calc_nb_score3_2()
        if self.w_v==0:
            return self.calc_nb_score_for_new_proposal_method_2_3_wv0()

        v_count=len(self.query_table) # 下限値の導出に利用. 1サブグラフあたりのラベルがデータのペアの数.

        count=0
        nb_count=0
        lib_list1=self.query_lib

        subgraph_id={}
        subgraph_score={}
        subgraph_nb_name={}
        subgraph_score_appr={}
        graph_id=0
        
        for nb_name in self.ans_list: # コード，ライブラリ，出力の類似度を計算
            lib_list2 = self.fetch_all_library_from_db(nb_name)
            lib_score = self.calc_lib_score(lib_list1, lib_list2)
                #calc_lib_start_time = timeit.default_timer()
                #lib_score = self.jaccard_similarity_coefficient(lib_list1, lib_list2)
                #calc_lib_end_time = timeit.default_timer()
                #self.each_calc_time_sum["Library"]+=calc_lib_end_time-calc_lib_start_time

            for subgraph in self.ans_list[nb_name]:
                graph_id+=1
                current_score=0
                v_score_appr=0
                current_score += lib_score * self.w_l

                for n_tuple in subgraph:
                    n1_name=n_tuple[0]
                    n2_name=n_tuple[1]
                    current_score+=self.calc_rel_c_o_with_timecount(n1_name, n2_name)
            
                #logging.info("score is ", current_score)
                #subgraph_score[subgraph]=current_score
                subgraph_id[graph_id]=subgraph
                subgraph_score[graph_id]=current_score
                subgraph_score_appr[graph_id]=current_score + v_score_appr
                subgraph_nb_name[graph_id]=nb_name


        if self.flg_running_faster["flg_optimize_calc_order"]:
            sorted_subgraph=sorted(subgraph_score.items(), key=lambda d: d[1], reverse=True)
        else:
            sorted_subgraph=list(subgraph_score.items())


        for pair in sorted_subgraph:
            graph_id=pair[0]
            #current_score = pair[1]
            current_score = subgraph_score[graph_id]
            nb_name=subgraph_nb_name[graph_id]
            
            if len(self.nb_score)<self.k:
                k_score=0
            else:
                self.top_k_score=sorted(self.nb_score.items(), key=lambda d: d[1], reverse=True)
                k_score=self.top_k_score[self.k-1][1]
                
            if self.flg_prune_under_sim_for_new_proposal_method(v_count, current_score, k_score):
                continue

            if nb_name not in self.nb_score:
                self.nb_score[nb_name]=0
            elif self.flg_prune_under_sim_for_new_proposal_method(v_count, current_score, self.nb_score[nb_name]):
                continue

            subgraph=subgraph_id[graph_id]
            remain_v_count=v_count
            for n_tuple in subgraph:
                n1_name=n_tuple[0]
                n2_name=n_tuple[1]
                v_score, remain_v_count=self.calc_rel_v_with_remain_count_with_timecount(n1_name, n2_name, remain_v_count)
                current_score+=v_score
                if remain_v_count==0 or self.flg_prune_under_sim_for_new_proposal_method(remain_v_count, current_score, max(k_score, self.nb_score[nb_name])):
                    break

            #logging.info("score is ", current_score)
            if self.nb_score[nb_name] < current_score:
                self.nb_score[nb_name]=current_score
            count+=1 #あとで消す
        nb_count+=1 #あとで消す

        return count, nb_count 

    # 使用 現状最速
    def calc_nb_score_for_proposal_method_with_juneau(self, w_c:float=None, w_v:float=None, w_l:float=None, w_d:float=None, k_juneau=20):
        """
        juneauがある場合の最適化．
        データ+出力+ライブラリ→コードの順に計算順序を最適化．
        """
        self.nb_score={}
        for w in [w_c, w_v, w_l, w_d]:
            if not w is None:
                self.set_each_w(w_c, w_v, w_l, w_d)


        if not self.flg_running_faster["flg_prune_under_sim"]:
            # For micro-benchmark.
            return self.calc_nb_score3_2()
        if self.w_v==0:
            return self.calc_nb_score_for_new_proposal_method_2_3_wv0()

        #juneau準備
        self.init_query_col()
        self.set_all_querytable_col()
        self.init_each_calc_time_sum()
        self.calc_all_tables_rel_using_juneau(k_juneau=k_juneau) #追加
        self.init_arranged_all_weights()

        #v_count=len(self.query_table) # 下限値の導出に利用. 1サブグラフあたりのラベルがデータのペアの数.
        c_count=len(self.query_cell_code) # 下限値の導出に利用. 1サブグラフあたりのラベルがデータのペアの数.

        count=0
        nb_count=0
        lib_list1=self.query_lib

        subgraph_id={}
        subgraph_score={}
        subgraph_nb_name={}
        subgraph_score_appr={}
        graph_id=0
        
        for nb_name in self.ans_list: # コード，ライブラリ，出力の類似度を計算
            lib_list2 = self.fetch_all_library_from_db(nb_name)
            lib_score = self.calc_lib_score(lib_list1, lib_list2)
            for subgraph in self.ans_list[nb_name]:
                graph_id+=1
                current_score=0
                v_score_appr=0
                current_score += lib_score * self.w_l

                for n_tuple in subgraph:
                    n1_name=n_tuple[0]
                    n2_name=n_tuple[1]
                    if self.is_data_match(nodenameQ=n1_name, nodenameN=n2_name):
                        sim = self.loadContentsSim(n1_name, n2_name)
                        if sim is not None: #juneauで計算した類似度以外のデータ類似度はゼロにする
                            current_score+= self.w_v * sim
                    #current_score+=self.calc_rel_c_o_with_timecount(n1_name, n2_name)
                    else:
                        current_score+=self.calc_rel_o_with_timecount(n1_name, n2_name)
            
                subgraph_id[graph_id]=subgraph
                subgraph_score[graph_id]=current_score
                subgraph_score_appr[graph_id]=current_score + v_score_appr
                subgraph_nb_name[graph_id]=nb_name


        if self.flg_running_faster["flg_optimize_calc_order"]:
            sorted_subgraph=sorted(subgraph_score.items(), key=lambda d: d[1], reverse=True)
        else:
            sorted_subgraph=list(subgraph_score.items())


        for pair in sorted_subgraph:
            graph_id=pair[0]
            #current_score = pair[1]
            current_score = subgraph_score[graph_id]
            nb_name=subgraph_nb_name[graph_id]
            
            if len(self.nb_score)<self.k:
                k_score=0
            else:
                self.top_k_score=sorted(self.nb_score.items(), key=lambda d: d[1], reverse=True)
                k_score=self.top_k_score[self.k-1][1]
                
            if self.flg_prune_under_sim_for_new_proposal_method_3(self.w_c, c_count, current_score, k_score):
                continue

            if nb_name not in self.nb_score:
                self.nb_score[nb_name]=0
            elif self.flg_prune_under_sim_for_new_proposal_method_3(self.w_c, c_count, current_score, self.nb_score[nb_name]):
                continue

            subgraph=subgraph_id[graph_id]
            remain_c_count=c_count
            for n_tuple in subgraph:
                n1_name=n_tuple[0]
                n2_name=n_tuple[1]
                #v_score, remain_v_count=self.calc_rel_v_with_remain_count_with_timecount(n1_name, n2_name, remain_v_count)
                c_score, remain_c_count=self.calc_rel_c_with_remain_number_with_timecount(n1_name, n2_name, remain_c_count)
                current_score+=c_score
                if remain_c_count==0 or self.flg_prune_under_sim_for_new_proposal_method_3(self.w_c, remain_c_count, current_score, max(k_score, self.nb_score[nb_name])):
                    break

            #logging.info("score is ", current_score)
            if self.nb_score[nb_name] < current_score:
                self.nb_score[nb_name]=current_score
            count+=1 #あとで消す
        nb_count+=1 #あとで消す

        return count, nb_count 

    def init_arranged_all_weights(self):
        if len(self.query_cell_code) != 0:
            self.w_c = self.w_c/len(self.query_cell_code)
        if len(self.query_table) != 0:
            self.w_v = self.w_v/len(self.query_table)
        output_num = 0
        for k,v in self.query_workflow_info["Display_data"].items():
            output_num+=v
        if output_num!=0:
            self.w_d = self.w_d/output_num

    def init_arranged_dataandoutput_weights(self):
        if len(self.query_table) != 0:
            self.w_v = self.w_v/len(self.query_table)
        output_num = 0
        for k,v in self.query_workflow_info["Display_data"].items():
            output_num+=v
        if output_num!=0:
            self.w_d = self.w_d/output_num

    # 使用 現状最速
    def calc_nb_score_for_new_proposal_method_2_3_wv0(self) -> Tuple[int, int]:
        """
        Call this function for the similairty calculation component of the proposed method if the data sim weight is zero.
        """
        c_count=len(self.query_cell_code) # 下限値の導出に利用. 1サブグラフあたりのラベルがデータのペアの数.

        count=0 #あとで消す
        nb_count=0 #あとで消す
        lib_list1=self.query_lib

        subgraph_id={}
        subgraph_score={}
        subgraph_nb_name={}
        graph_id=0
        
        for nb_name in self.ans_list: # コード，ライブラリ，出力の類似度を計算
            lib_list2 = self.fetch_all_library_from_db(nb_name)
            lib_score = self.calc_lib_score(lib_list1, lib_list2)

            for subgraph in self.ans_list[nb_name]:
                graph_id+=1
                current_score=0
                current_score += lib_score * self.w_l

                for n_tuple in subgraph:
                    n1_name=n_tuple[0]
                    n2_name=n_tuple[1]
                    current_score+=self.calc_rel_o_with_timecount(n1_name, n2_name)

                subgraph_id[graph_id]=subgraph
                subgraph_score[graph_id]=current_score
                subgraph_nb_name[graph_id]=nb_name


        if self.flg_running_faster["flg_optimize_calc_order"]:
            sorted_subgraph=sorted(subgraph_score.items(), key=lambda d: d[1], reverse=True)
        else:
            sorted_subgraph=list(subgraph_score.items())

        for pair in sorted_subgraph:
            graph_id=pair[0]
            current_score = pair[1]
            nb_name=subgraph_nb_name[graph_id]
            
            if len(self.nb_score)<self.k:
                k_score=0
            else:
                self.top_k_score=sorted(self.nb_score.items(), key=lambda d: d[1], reverse=True)
                k_score=self.top_k_score[self.k-1][1]
                
            if self.flg_prune_under_sim_for_new_proposal_method_3(self.w_c, c_count, current_score, k_score):
                continue

            if nb_name not in self.nb_score:
                self.nb_score[nb_name]=0
            elif self.flg_prune_under_sim_for_new_proposal_method_3(self.w_c, c_count, current_score, self.nb_score[nb_name]):
                continue

            subgraph=subgraph_id[graph_id]
            remain_c_count=c_count
            for n_tuple in subgraph:
                n1_name=n_tuple[0]
                n2_name=n_tuple[1]
                c_score, remain_c_count=self.calc_rel_c_with_remain_number_with_timecount(n1_name, n2_name, remain_c_count)
                current_score+=c_score
                if remain_c_count==0 or self.flg_prune_under_sim_for_new_proposal_method_3(self.w_c, remain_c_count, current_score, max(k_score, self.nb_score[nb_name])):
                    break

            #logging.info("score is ", current_score)
            if self.nb_score[nb_name] < current_score:
                self.nb_score[nb_name]=current_score
            count+=1 #あとで消す
        nb_count+=1 #あとで消す

        return count, nb_count #あとで消す

    #使用
    def calc_lib_score(self, lib_list1:list, lib_list2:list) -> float:
        """
        Calculate the library similarity as Jaccard similarity coefficient.

        Args:
            lib_list1 (list[str]): a list of name of libraries given as the query.
            lib_list2 (list[str]): a list of name of libraries used in a computational notebook.

        Returns:
            float: library similarity between given libraries lists.
        """
        calc_lib_start_time = timeit.default_timer()
        lib_score = self.jaccard_similarity_coefficient(lib_list1, lib_list2)
        calc_lib_end_time = timeit.default_timer()
        self.each_calc_time_sum["Library"]+=calc_lib_end_time-calc_lib_start_time
        return lib_score

    def top_k_nb(self, k:int) -> list:
        """
        Args:
            k (int): a natural number for top-k search on computational notebooks.
        Returns:
            list: a sorted list of top-k computational notebooks with each similarity.
        """
        sorted_nb_score = sorted(self.nb_score.items(), key=lambda x:x[1], reverse=True)
        k_tmp = min(len(sorted_nb_score), k)
        ret=sorted_nb_score[:k_tmp]
        self.top_k_score=ret
        return ret
           
    def fetch_source_code_table(self, cleaned_nb_name:str) -> pd.DataFrame:
        """
        Fetch the table of source codes and return it in form of DataFrame.
        """
        if cleaned_nb_name in self.cell_source_code:
            return self.cell_source_code[cleaned_nb_name]
        
        if self.flg_use_artifical_dataset and self.is_artifical_dataset(cleaned_nb_name):
            load_name = cleaned_nb_name[:cleaned_nb_name.rfind("cp")]
        else:
            load_name  = cleaned_nb_name

        
        with self.postgres_eng.connect() as conn:
            try:
                code_table = pd.read_sql_table(f"cellcont_{load_name}", conn, schema=f"{config.sql.cellcode}")
                self.cell_source_code[cleaned_nb_name]=code_table
            except:
                logging.error(f"error: collecting source code from db is failed.")
        return self.cell_source_code[cleaned_nb_name]
        
    def fetch_var_table(self, table_name:str) -> pd.DataFrame:
        """
        Fetch the table of source codes and return it in form of DataFrame.

        Args:
            table_name (str): String of f'rtable{cell id}_{variable name}_{notebook name}'. f'rtable{セル番号}_{変数名}_{NB名}'の文字列．
        
        Returns:
            pd.DataFrame:
        """

        if self.flg_use_artifical_dataset and self.is_artifical_dataset(table_name):
            load_name = table_name[:table_name.rfind("cp")]
        else:
            load_name  = table_name

        with self.postgres_eng.connect() as conn:
            try:
                var_table = pd.read_sql_table(f"rtable{load_name}", conn, schema=f"{config.sql.dbs}")
                return var_table
            except Exception as e:
                logging.error(f"error: collecting var table '{load_name}' from db is failed because of {e}")
                return None

    def bench_mark_calc_cell_rel(self, cleaned_nb_name_A, cleaned_nb_name_B, ver="jaccard_similarity_coefficient"):
        """
        NB名がcleaned_nb_name_Aとcleaned_nb_name_Bの2つのNBに対して総当たりでセルのコードの類似度を計算するベンチマーク．
        参考論文: Clone Detection Using Abstract Syntax Suffix Trees
        参考論文のgithub: https://github.com/panchdevs/code-detection
        """
        time1=0
        calc_count=0
        code_table_A=self.fetch_source_code_table(cleaned_nb_name_A)
        code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
        class_code_relatedness=CodeRelatedness()
        for cell_id_A in code_table_A["cell_id"].values:
            for cell_id_B in code_table_B["cell_id"].values:
                #print(cell_id_A, " ", cell_id_B)
                code_A=code_table_A[code_table_A["cell_id"]==cell_id_A]["cell_code"].values
                code_B=code_table_B[code_table_B["cell_id"]==cell_id_B]["cell_code"].values
                code_A="".join(list(code_A))
                code_B="".join(list(code_B))
                #print(code_A)
                #print(code_B)
                #comparer = CodeComparer(codebase_path=None)
                #matches = comparer.compare(code_body)
                #self.calc_cell_rel_old(cleaned_nb_name_A, cleaned_nb_name_B, cell_id_A, cell_id_B)
                
                if ver=="jaccard_similarity_coefficient":
                    start_time1 = timeit.default_timer()
                    rel=class_code_relatedness.calc_code_rel_by_jaccard_index(code_A, code_B)
                    #print(rel)
                    end_time1 = timeit.default_timer()
                    time1+=end_time1-start_time1
                    calc_count+=1
                elif ver=="hash_jaccard_similarity_coefficient":
                    start_time1 = timeit.default_timer()
                    rel=class_code_relatedness.calc_code_rel_by_hash_and_jaccard_index(code_A, code_B)
                    #print(rel)
                    end_time1 = timeit.default_timer()
                    time1+=end_time1-start_time1
                    calc_count+=1
                elif ver=="graph_edit_distance":
                    start_time1 = timeit.default_timer()
                    rel=self.calc_cell_rel_by_graph_edit_dist2(cleaned_nb_name_A, cleaned_nb_name_B, cell_id_A, cell_id_B)
                    if rel!=0:
                        #print(rel)
                        pass
                    end_time1 = timeit.default_timer()
                    time1+=end_time1-start_time1
                    calc_count+=1
                #print(cell_id_A, " ", cell_id_B, " ended")
                #break #あとで消す
            #break #あとで消す
        return time1, calc_count


    def calc_cell_rel_old(self, cleaned_nb_name_A, cleaned_nb_name_B, cell_id_A, cell_id_B, ver="jaccard_similarity_coefficient"):
        """
        参考論文: Clone Detection Using Abstract Syntax Suffix Trees
        参考論文のgithub: https://github.com/panchdevs/code-detection
        """
        code_table_A=self.fetch_source_code_table(cleaned_nb_name_A)
        code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
        if (not cell_id_A in code_table_A["cell_id"].values) or (not cell_id_B in code_table_B["cell_id"].values):
            return 0.0
        code_A=code_table_A[code_table_A["cell_id"]==cell_id_A]["cell_code"].values
        code_B=code_table_B[code_table_B["cell_id"]==cell_id_B]["cell_code"].values
        code_A="".join(list(code_A))
        code_B="".join(list(code_B))
        #print(code_A)
        #print(code_B)
        #comparer = CodeComparer(codebase_path=None)
        #matches = comparer.compare(code_body)
        #self.calc_cell_rel_old(cleaned_nb_name_A, cleaned_nb_name_B, cell_id_A, cell_id_B)
        
        class_code_relatedness=CodeRelatedness()
        if ver=="jaccard_similarity_coefficient":
            return class_code_relatedness.calc_code_rel_by_jaccard_index(code_A, code_B)
        elif ver=="hash_jaccard_similarity_coefficient":
            return class_code_relatedness.calc_code_rel_by_hash_and_jaccard_index(code_A, code_B)
        elif ver=="graph_edit_distance":
            return self.calc_cell_rel_by_graph_edit_dist2(cleaned_nb_name_A, cleaned_nb_name_B, cell_id_A, cell_id_B)
        pass

    @timeout_decorator.timeout(30)
    def generate_graph_with_timeout(self, graph):
        return generate_graph(graph)

    @timeout_decorator.timeout(30)
    def parse_code_with_timeout(self, code):
        return self.__parse_code(code)
        
    @timeout_decorator.timeout(30)
    def graph_edit_distance_with_timeout(self, g_A, g_B):
        return nx.graph_edit_distance(g_A, g_B)

    def calc_cell_rel_by_graph_edit_dist2(self, cleaned_nb_name_A, cleaned_nb_name_B, cell_id_A, cell_id_B):
        """
        グラフの編集距離をライブラリnetworkxの関数networkx.graph_edit_distance(graphA, graphB)で計算．
        ノード数が多い方のグラフのノード数で編集距離を割ることで正規化(0以上1以下にする)，
        その値を1から引くことでコードの関連度スコア(高い方が類似したコード)とする．

        Returns:
            float-like: コードの関連度
        """
        code_table_A=self.fetch_source_code_table(cleaned_nb_name_A)
        code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
        if (not cell_id_A in code_table_A["cell_id"].values) or (not cell_id_B in code_table_B["cell_id"].values):
            return 0.0
        code_A=code_table_A[code_table_A["cell_id"]==cell_id_A]["cell_code"].values
        code_B=code_table_B[code_table_B["cell_id"]==cell_id_B]["cell_code"].values
        try:
            #depA, line2cid_A, all_code_A =self.parse_code_with_timeout(code_A[0].split("\n"))
            depA, line2cid_A, all_code_A =self.parse_code_with_timeout(code_A)
            #depB, line2cid_B, all_code_B =self.parse_code_with_timeout(code_B[0].split("\n"))
            depB, line2cid_B, all_code_B =self.parse_code_with_timeout(code_B)
        except:
            logging.info(f"timeout: get dependency")
            logging.info(f"code_A is...")
            logging.info(f"{code_A}")
            logging.info(f"code_B is...")
            logging.info(f"{code_B}")
            return 0.0
        #print(depA)
        #print(depB)

        try:
            g_A=self.generate_graph_with_timeout(depA)
            self.g_A=g_A
            g_B=self.generate_graph_with_timeout(depB)
            self.g_B=g_B
        except:
            logging.info(f"timeout: build dependency graph")
            logging.info(f"code_A is...")
            logging.info(f"{code_A}")
            logging.info(f"dep_A is...")
            logging.info(f"{depA}")
            logging.info(f"code_B is...")
            logging.info(f"{code_B}")
            logging.info(f"dep_B is...")
            logging.info(f"{depB}")
            return 0.0

        if min(len(list(g_A.nodes)), len(list(g_B.nodes)))==0:
            return 0.0

        try:
            graph_edit_dist=self.graph_edit_distance_with_timeout(g_A, g_B)
        except:
            logging.info(f"timeout: calc graph edit distance")
            logging.info(f"code_A is...")
            logging.info(f"{code_A}")
            logging.info(f"code_B is...")
            logging.info(f"{code_B}")
            logging.info(f"dep_A is...")
            logging.info(f"{depA}")
            nx.draw_networkx(g_A)
            plt.show
            logging.info(f"dep_B is...")
            logging.info(f"{depB}")
            nx.draw_networkx(g_B)
            plt.show
            return 0.0

        self.normalization=max(len(g_A.nodes), len(g_B.nodes)) #スコア正規化のため割り算の分母
        ret_score=1-(graph_edit_dist/self.normalization)
        return ret_score


    def calc_cell_rel_by_graph_edit_dist(self, cleaned_nb_name_A, cleaned_nb_name_B, cell_id_A, cell_id_B):
        """
        グラフの編集距離をライブラリnetworkxの関数networkx.graph_edit_distance(graphA, graphB)で計算．
        ノード数が多い方のグラフのノード数で編集距離を割ることで正規化(0以上1以下にする)，
        その値を1から引くことでコードの関連度スコア(高い方が類似したコード)とする．

        Returns:
            float-like: コードの関連度
        """
        code_table_A=self.fetch_source_code_table(cleaned_nb_name_A)
        code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
        if (not cell_id_A in code_table_A["cell_id"].values) or (not cell_id_B in code_table_B["cell_id"].values):
            return 0.0
        code_A=code_table_A[code_table_A["cell_id"]==cell_id_A]["cell_code"].values
        code_B=code_table_B[code_table_B["cell_id"]==cell_id_B]["cell_code"].values
        try:
            depA, line2cid_A, all_code_A =self.__parse_code(code_A)
            depB, line2cid_B, all_code_B =self.__parse_code(code_B)
        except:
            logging.info(f"timeout: calc graph edit distance")
            logging.info(f"code_A is...")
            logging.info(f"{code_A}")
            logging.info(f"code_B is...")
            logging.info(f"{code_B}")
        try:
            g_A=generate_graph(depA)
            g_B=generate_graph(depB)
        except:
            logging.info(f"timeout: calc graph edit distance")
            logging.info(f"code_A is...")
            logging.info(f"{code_A}")
            logging.info(f"code_B is...")
            logging.info(f"{code_B}")
        if min(len(g_A.nodes), len(g_B.nodes))==0:
            return 0.0
        try:
            graph_edit_dist=nx.graph_edit_distance(g_A, g_B)
        except:
            logging.info(f"timeout: calc graph edit distance")
            logging.info(f"code_A is...")
            logging.info(f"{code_A}")
            logging.info(f"code_B is...")
            logging.info(f"{code_B}")

        self.normalization=max(len(g_A.nodes), len(g_B.nodes)) #スコア正規化のため割り算の分母
        ret_score=1-(graph_edit_dist/self.normalization)
        return ret_score


    #不使用になりそう
    def calc_table_rel(self, tableA:pd.DataFrame, tableB:pd.DataFrame) -> float:
        """
        引数で与えた2テーブルに対し，ジャカード係数を元に類似度を計算する．

        バグ修正版．
        バグ：同じ列名がtableAとtableBに含まれているときacol_setのキャッシュが原因でエラーが発生．

        Args:
            tableA (DataFrame)
            tableB (DataFrame)
        
        Returns:
            float-like: tableAとtableBの類似度．
        """
        if self.flg_juneau:
            return  0.0
        #group idはnb_nameで代用できる(つながっているワークフローごとに一意なので)
        #logging.info("calc table sim...")

        rel_score=0.0
        matching=[]
        acol_set_A = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        acol_set_B = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名A: 列名B}}
        r_matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名B: 列名A}}

        scmaA = tableA.columns.values # scmaA (list[str]): tableAの列名リスト
        scmaB = tableB.columns.values # scmaB (list[str]): tableBの列名リスト
        for nameA in scmaA:
            if nameA == "Unnamed: 0" or "index" in nameA or nameA=="Unnamed:0":
                continue

            start_time_set = timeit.default_timer()
            if nameA not in acol_set_A: #以前にデータ値を処理したものは取っておく
                colA = tableA[nameA][~pd.isnull(tableA[nameA])].values # ~はビット反転演算子
                acol_set_A[nameA] = list(set(colA)) # 重複無しリスト
            else:
                colA=acol_set_A[nameA]
            end_time_set = timeit.default_timer()
            self.calc_v_microbenchmark["set"]+=end_time_set-start_time_set

            for nameB in scmaB:
                if nameB == "Unnamed: 0" or "index" in nameB or nameB=="Unnamed:0":
                    continue
                if tableA[nameA].dtype is not tableB[nameB].dtype: # 列nameAと列nameBのデータ型が異なる場合
                    continue

                start_time_set = timeit.default_timer()
                if nameB not in acol_set_B:
                    colB = tableB[nameB][~pd.isnull(tableB[nameB])].values # ~はビット反転演算子
                    acol_set_B[nameB] = list(set(colB)) # 重複無しリスト
                else:
                    colB=acol_set_B[nameB]
                end_time_set = timeit.default_timer()
                self.calc_v_microbenchmark["set"]+=end_time_set-start_time_set

                start_time_colmun_rel = timeit.default_timer()
                sim_col = self.jaccard_similarity_coefficient(colA, colB)
                end_time_colmun_rel = timeit.default_timer()
                self.calc_v_microbenchmark["column_rel"]+=end_time_colmun_rel-start_time_colmun_rel
                matching.append((nameA, nameB, sim_col)) # list[str, str, float]: [列名A, 列名B, 列Aと列Bの類似度]

        start_time_sort = timeit.default_timer()
        matching = sorted(matching, key=lambda d: d[2], reverse=True) # 類似度が高い順にソート
        end_time_sort = timeit.default_timer()
        self.calc_v_microbenchmark["sort"]+=end_time_sort-start_time_sort
    
        start_time_table_rel = timeit.default_timer()
        count=0
        max_count=min(len(scmaA), len(scmaB))
        if max_count==0:
            return 0.0

        for i in range(len(matching)):
            if count>=max_count:
                break

            if matching[i][0] not in matched_pair and matching[i][1] not in r_matched_pair:
                matched_pair[matching[i][0]]=matching[i][1]
                r_matched_pair[matching[i][1]]=matching[i][0]
                rel_score+=matching[i][2]
                count+=1
        rel_score=rel_score/max(len(scmaA), len(scmaB))
        end_time_table_rel = timeit.default_timer()
        self.calc_v_microbenchmark["table_rel"]+=end_time_table_rel-start_time_table_rel
        return rel_score

    #不使用になりそう
    def calc_table_rel_between_query_and_db(self, query_node_name:str, tableB:pd.DataFrame) -> float:
        """
        引数で与えた2テーブルに対し，ジャカード係数を元に類似度を計算する．

        バグ修正版．
        バグ：同じ列名がtableAとtableBに含まれているときacol_setのキャッシュが原因でエラーが発生．

        Args:
            tableB (DataFrame)
        
        Returns:
            float-like: tableAとtableBの類似度．
        """
        if self.flg_juneau:
            return 0.0
        #group idはnb_nameで代用できる(つながっているワークフローごとに一意なので)
        #logging.info("calc table sim...")
        rel_score=0.0
        matching=[]
        acol_set_A = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        acol_set_B = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名A: 列名B}}
        r_matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名B: 列名A}}
        
        scmaB = tableB.columns.values # scmaB (list[str]): tableBの列名リスト

        #テスト実行用
        #query_table_name_list=list(self.query_table.keys())
        #query_table_name=query_table_name_list[0]
        #for name, scma in self.query_scma_dict.items():
        #    if scma is scmaA:
        #        query_table_name=name
        #        break
        tableA=self.query_table[query_node_name]
        scmaA = tableA.columns.values
        if query_node_name in self.query_acol_set:
            acol_set_A=self.query_acol_set[query_node_name]

        for nameA in scmaA:
            if nameA == "Unnamed: 0" or "index" in nameA or nameA=="Unnamed:0":
                continue

            if nameA not in acol_set_A: #以前にデータ値を処理したものは取っておく
                colA = tableA[nameA][~pd.isnull(tableA[nameA])].values # ~はビット反転演算子
                acol_set_A[nameA] = list(set(colA)) # 重複無しリスト
            else:
                colA=acol_set_A[nameA]

            for nameB in scmaB:
                if nameB == "Unnamed: 0" or "index" in nameB or nameB=="Unnamed:0":
                    continue
                if tableA[nameA].dtype is not tableB[nameB].dtype: # 列nameAと列nameBのデータ型が異なる場合
                    continue

                if nameB not in acol_set_B:
                    colB = tableB[nameB][~pd.isnull(tableB[nameB])].values # ~はビット反転演算子
                    acol_set_B[nameB] = list(set(colB)) # 重複無しリスト
                else:
                    colB=acol_set_B[nameB]

                sim_col = self.jaccard_similarity_coefficient(colA, colB)
                matching.append((nameA, nameB, sim_col)) # list[str, str, float]: [列名A, 列名B, 列Aと列Bの類似度]
        
        if query_node_name not in self.query_acol_set:
            self.query_acol_set[query_node_name]=acol_set_A

        matching = sorted(matching, key=lambda d: d[2], reverse=True) # 類似度が高い順にソート
    
        count=0
        max_count=min(len(scmaA), len(scmaB))
        if max_count==0:
            return 0.0

        for i in range(len(matching)):
            if count>=max_count:
                break

            if matching[i][0] not in matched_pair and matching[i][1] not in r_matched_pair:
                matched_pair[matching[i][0]]=matching[i][1]
                r_matched_pair[matching[i][1]]=matching[i][0]
                rel_score+=matching[i][2]
                count+=1
        rel_score=rel_score/max(len(scmaA), len(scmaB))
        return rel_score

    def inner_calc_data_sim(self, query_node_name:str, tnameN:str) -> float:
        """
        引数で与えた2テーブルに対し，ジャカード係数を元に類似度を計算する．

        バグ修正版．
        バグ：同じ列名がtableAとtableBに含まれているときacol_setのキャッシュが原因でエラーが発生．

        Args:
            tableB (DataFrame)
        
        Returns:
            float-like: tableAとtableBの類似度．
        """
        #引数を変えた版のcalc_table_rel_between_query_and_db()．

        start_time_load=timeit.default_timer()
        tableA=self.query_table[query_node_name]
        tableB=self.fetch_var_table(tnameN)
        end_time_load=timeit.default_timer()
        self.calc_v_microbenchmark["load"]+=end_time_load-start_time_load

        if tableB is None:
            return 0.0
        #group idはnb_nameで代用できる(つながっているワークフローごとに一意なので)
        #logging.info("calc table sim...")
        rel_score=0.0
        matching=[]
        acol_set_A = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        acol_set_B = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名A: 列名B}}
        r_matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名B: 列名A}}
        
        scmaB = tableB.columns.values # scmaB (list[str]): tableBの列名リスト

        #テスト実行用
        #query_table_name_list=list(self.query_table.keys())
        #query_table_name=query_table_name_list[0]
        #for name, scma in self.query_scma_dict.items():
        #    if scma is scmaA:
        #        query_table_name=name
        #        break
        scmaA = tableA.columns.values
        if query_node_name in self.query_acol_set:
            acol_set_A=self.query_acol_set[query_node_name]

        for nameA in scmaA:
            if nameA == "Unnamed: 0" or "index" in nameA or nameA=="Unnamed:0":
                continue

            if nameA not in acol_set_A: #以前にデータ値を処理したものは取っておく
                colA = tableA[nameA][~pd.isnull(tableA[nameA])].values # ~はビット反転演算子
                acol_set_A[nameA] = list(set(colA)) # 重複無しリスト
            else:
                colA=acol_set_A[nameA]

            for nameB in scmaB:
                if nameB == "Unnamed: 0" or "index" in nameB or nameB=="Unnamed:0":
                    continue
                if tableA[nameA].dtype is not tableB[nameB].dtype: # 列nameAと列nameBのデータ型が異なる場合
                    continue

                if nameB not in acol_set_B:
                    colB = tableB[nameB][~pd.isnull(tableB[nameB])].values # ~はビット反転演算子
                    acol_set_B[nameB] = list(set(colB)) # 重複無しリスト
                else:
                    colB=acol_set_B[nameB]

                sim_col = self.jaccard_similarity_coefficient(colA, colB)
                matching.append((nameA, nameB, sim_col)) # list[str, str, float]: [列名A, 列名B, 列Aと列Bの類似度]
        
        if query_node_name not in self.query_acol_set:
            self.query_acol_set[query_node_name]=acol_set_A

        matching = sorted(matching, key=lambda x: x[2], reverse=True) # 類似度が高い順にソート
    
        count=0
        max_count=min(len(scmaA), len(scmaB))
        if max_count==0:
            return 0.0

        for i in range(len(matching)):
            if count>=max_count:
                break

            if matching[i][0] not in matched_pair and matching[i][1] not in r_matched_pair:
                matched_pair[matching[i][0]]=matching[i][1]
                r_matched_pair[matching[i][1]]=matching[i][0]
                rel_score+=matching[i][2]
                count+=1
        rel_score=rel_score/max(len(scmaA), len(scmaB))
        return rel_score

    def calc_table_rel_approximately(self, tableA, tableB):
        """
        データタイプのみで関連度を概算．
        """
        #group idはnb_nameで代用できる(つながっているワークフローごとに一意なので)
        #logging.info("calc table sim...")

        rel_score=0.0
        matching=[]
        acol_set_A = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        acol_set_B = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名A: 列名B}}
        r_matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名B: 列名A}}

        scmaA = tableA.columns.values # scmaA (list[str]): tableAの列名リスト
        scmaB = tableB.columns.values # scmaB (list[str]): tableBの列名リスト
        
        # -------- 後で分離 ----------------
        self.type_dict_A={}
        for nameA in scmaA:
            if nameA == "Unnamed: 0" or "index" in nameA or nameA=="Unnamed:0":
                continue
            typeA=tableA[nameA].dtype
            if typeA not in self.type_dict_A:
                self.type_dict_A[typeA]=0
            self.type_dict_A[typeA]+=1
        # ---------------------------------
            
            
        type_dict_B={}
        for nameB in scmaB:
            if nameB == "Unnamed: 0" or "index" in nameB or nameB=="Unnamed:0":
                continue
            typeB=tableB[nameB].dtype
            if typeB not in type_dict_B:
                type_dict_B[typeB]=0
            type_dict_B[typeB]+=1

        rel_score=0
        allA=0
        #クエリのテーブルに対し，データタイプの対応がとれる列の割合を計算
        for typeA in self.type_dict_A:
            allA+=self.type_dict_A[typeA]
            if typeA not in type_dict_B:
                continue
            if self.type_dict_A[typeA] < type_dict_B[typeA]:
                rel_score+=self.type_dict_A[typeA]
                continue
            rel_score+=type_dict_B[typeA]
        if allA == 0:
            return 0
        rel_score=rel_score/allA
        #sim_col = self.jaccard_similarity_coefficient(colA, colB)
        #rel_score=rel_score/max(len(scmaA), len(scmaB))
        return rel_score

    def set_group_id(self, nb_name):
        if len(self.nb_name_group_id)==0:
            self.nb_name_group_id[nb_name]=0
        if nb_name not in self.nb_name_group_id:
            max_gid=0
            for gid in self.nb_name_group_id.values:
                max_gid = max(max_gid, gid)
            self.nb_name_group_id[nb_name]=max_gid+1


    def fetch_all_library_from_db(self, cleaned_nb_name:str) -> list:# -> List[str]:
        """
        Cache library names of the all computational notebooks in database, 
        and return a list of library names used in 'cleaned_nb_name' computational notebooks.
        """
        if cleaned_nb_name in self.library:
            return self.library[cleaned_nb_name]
        logging.info("collecting library info...")
        with self.postgres_eng.connect() as conn:
            try:
                lib_table = pd.read_sql_table("nb_libraries", conn, schema=f"{config.sql.nb_info}")
                for _, row in lib_table.iterrows():
                    nb_name=row["nb_name"]
                    lib=row["libraries"]
                    if nb_name not in self.library:
                        self.library[nb_name]=[]
                    if lib not in self.library[nb_name]:
                        self.library[nb_name].append(lib)
                logging.info("collecting library info : completed.")
            except:
                logging.info("collecting library info : failed.")
        return self.library[cleaned_nb_name]
        
    def fetch_multiset_of_display_type_from_db(self, cleaned_nb_name:str) -> list:# -> List[str]:
        """
        Returns:
            list(str): multiset of display data type that the nb outputs.
        """
        if cleaned_nb_name in self.display_type:
            return self.display_type[cleaned_nb_name]
        logging.info("collecting display type info...")
        with self.postgres_eng.connect() as conn:
            try:
                d_table = pd.read_sql_table("display_type", conn, schema=f"{config.sql.nb_info}")
                for _, row in d_table.iterrows():
                    nb_name=row["nb_name"]
                    t=row["data_type"]
                    if nb_name not in self.display_type:
                        self.display_type[nb_name]=[]
                    if t not in self.display_type[nb_name]:
                        self.display_type[nb_name].append(t)
                logging.info("collecting display type info : completed.")
            except:
                logging.info("collecting display type info : failed.")
        return self.display_type[cleaned_nb_name]


    def fetch_display_type_and_cell_id_from_db(self, cleaned_nb_name):
        """
        Returns:
            list[tuple(int-like, str)]: tuple is cell id and display data type in the cell
        """
        if cleaned_nb_name in self.display_type_and_cell_id:
            return self.display_type_and_cell_id[cleaned_nb_name]
        logging.info("collecting display type and cell id info...")
        with self.postgres_eng.connect() as conn:
            try:
                d_table = pd.read_sql_table("display_type_and_cell_id", conn, schema=f"{config.sql.nb_info}")
                for _, row in d_table.iterrows():
                    nb_name=row["nb_name"]
                    cid=row["cell_id"]
                    t=row["data_type"]
                    if nb_name not in self.display_type_and_cell_id:
                        self.display_type_and_cell_id[nb_name]=[]
                    if (cid, t) not in self.display_type_and_cell_id[nb_name]:
                        self.display_type_and_cell_id[nb_name].append((cid, t))
                logging.info("collecting display type and cell id info : completed.")
            except:
                logging.info("collecting display type and cell id info : failed.")
        return self.display_type_and_cell_id[cleaned_nb_name]

    # 提案手法で使用
    def flg_prune_under_sim(self, visited:List[str], current_score:float, max_c_score:float=1, max_v_score:float=1, max_d_score:float=1) -> bool:
        """
        Args:
            max_c_score (int or float or int-like): とりうるセル類似度の最大値
            max_v_score (int or float or int-like): とりうるテーブル類似度の最大値
        """
        if len(self.nb_score) < self.k:
            return False
        
        # k番目のスコアの値をセット
        self.top_k_score=sorted(self.nb_score.items(), key=lambda d: d[1], reverse=True)
        k_score=self.top_k_score[self.k-1][1]

        c_count=0
        v_count=0
        d_count=0
        # 毎回ループでチェックするのは遅そう
        if len(visited) == len(self.QueryGraph.nodes):
            under_limit = k_score - current_score
        else:
            for n in self.QueryGraph.nodes:
                if n in visited:
                    continue
                if self.attr_of_q_node_type[n]=="Cell":
                    c_count+=1
                elif self.attr_of_q_node_type[n]=="Var":
                    v_count+=1
                elif self.attr_of_q_node_type[n]=="Display_data":
                    d_count+=1
            under_limit = k_score - self.w_c * (c_count * max_c_score) - self.w_v * (v_count * max_v_score) - self.w_d * (d_count * max_d_score) - current_score
            #under_limit=max(0, under_limit)
        if under_limit>0:
            # 枝刈りの時
            return True
        else:
            return False

    # 新しい方の提案手法 関数new_proposal_method で使用
    def flg_prune_under_sim_for_new_proposal_method(self, v_count, current_score, compare_score, max_v_score=1):
        #if compare_score <=0:
        #    return True
        under_limit = compare_score - self.w_v * (v_count * max_v_score) - current_score
        if under_limit>0:
            # 枝刈りの時
            return True
        else:
            return False

    # 新しい方の提案手法 関数new_proposal_method で使用 （var以外にも対応）
    def flg_prune_under_sim_for_new_proposal_method_3(self, weight:float, remain_count:int, current_score:float, compare_score:float, max_score:float=1) -> bool:
        under_limit = compare_score - weight * (remain_count * max_score) - current_score
        if under_limit>0:
            return True
        else:
            return False

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

    @staticmethod
    def jaccard_similarity(self, colA, colB):
        return self.jaccard_similarity_coefficient(colA, colB)

    @staticmethod
    def row_similarity(colA:list, colB:list) -> float:
        # search/search_tables.pyからコピー
        colA_value = colA[~pd.isnull(colA)].values # null値は除去
        colB_value = colB[~pd.isnull(colB)].values
        row_sim_upper = len(np.intersect1d(colA_value, colB_value))
        row_sim_lower = len(np.union1d(colA_value, colB_value))
        row_sim = float(row_sim_upper) / float(row_sim_lower)
        return row_sim

    @staticmethod
    def col_similarity(tableA:pd.DataFrame, tableB:pd.DataFrame, SM:dict, key_factor:float) -> float:
        """
        Args:
            tableA : 表形式データ
            tableB : 表形式データ
            SM (dict): スキーママッピング(2つのテーブルのキー(列名)の対応関係)。おそらく{str: str}で、{tableAのキー: tableBのキー}。
            key_factor (float): key_scoreに相当
        
        Returns:
            float: 列の類似度
        """
        col_sim_upper = 1 + float(len(SM.keys()) - 1) * float(key_factor)
        tableA_not_in_tableB = [] # キーのリスト
        for kyA in tableA.columns.tolist():
            if kyA not in SM:
                tableA_not_in_tableB.append(kyA)
        col_sim_lower = len(tableB.columns.values) + len(tableA_not_in_tableB) # tableAとtableBのキーの重複無し集合
        col_sim = float(col_sim_upper) / float(col_sim_lower)
        return col_sim


    def previous_calc_output_sim(self, nb_name:str) -> float:
        """
        Calculate output similarity for set-based methods.
        """
        calc_outputs_start_time = timeit.default_timer()
        count_1=0
        count_2=0
        ele = "Display_data"
        for display_type in self.query_workflow_info[ele]:
            count_1+=self.query_workflow_info[ele][display_type]
            if display_type not in self.db_workflow_info[nb_name][ele]:
                count_2+=0
            elif self.query_workflow_info[ele][display_type] > self.db_workflow_info[nb_name][ele][display_type]:
                count_2+=self.db_workflow_info[nb_name][ele][display_type]
        if count_1==0:
            score=0
        else:
            score=count_2/count_1
        calc_outputs_end_time = timeit.default_timer()
        self.each_calc_time_sum["Display_data"]+=calc_outputs_end_time-calc_outputs_start_time
        return score

    def existing_method_calc_table_sim(self):
        """
        Calculate data similarity for the data-based search method.
        表データだけをみてtop-kのスコアにする．(self.nb_scoreにセット)
        ノートブック内の表データのうち，クエリと最も類似度が高くなる組み合わせでノートブックをスコアリングする．
        """
        sim_list={}
        n_q_num=len(self.query_table)
        #node_db_list=list(self.G.nodes)
        node_db_list=self.all_node_list["Var"]
        for n_q in self.query_table:
            for n_db in node_db_list:
                if self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    continue
                nb_name=self.attr_of_db_nb_name[n_db]
                if nb_name not in self.valid_nb_name:
                    continue
                sim=self.calc_rel_with_timecount(n_q, n_db)
                if nb_name not in sim_list:
                    sim_list[nb_name]={}
                    self.nb_score[nb_name]=0
                sim_list[nb_name][(n_q, n_db)]=sim
        self.sim_list=sim_list
        for nb_name in sim_list:
            if n_q_num > len(sim_list[nb_name]):
                self.nb_score[nb_name]=0
                continue
            sorted_list=sorted(sim_list[nb_name].items(), key=lambda d: d[1], reverse=True)
            visited_n_q=[]
            for i in range(min(n_q_num, len(sorted_list))):
                if sorted_list[i][0][0] in visited_n_q:
                    continue
                visited_n_q.append(sorted_list[i][0][0])
                self.nb_score[nb_name]+=sorted_list[i][1]

    def existing_method_calc_table_sim_using_juneau(self, k_juneau:int=20) -> dict:
        """
        1と2は結果を戻り値にするかself.nb_scoreに格納するかの違い．
        """
        nb_score={}
        sim_list={}
        n_q_num=len(self.query_table)
        #node_db_list=list(self.G.nodes)

        self.flg_juneau=True
        for tnameA, tableA in self.query_table.items():
            #tableA = self.fetch_var_table(nname)

            rtables=self.search_similar_tables_threshold2_with_scores(query=tableA, beta=0.1, k=k_juneau, theta=10, thres_key_cache=0.2, thres_key_prune=0.9, tflag=True)
            for tname, _, score in rtables:
                tgroup = self.table_group[tname]
                for tnameB, tgroupB in self.table_group.items():
                    if tgroup == tgroupB:
                        nb_name=self.attr_of_db_nb_name[tnameB]
                        if nb_name not in self.valid_nb_name:
                            continue
                        if nb_name not in sim_list:
                            sim_list[nb_name]={}
                            nb_score[nb_name]=0
                        sim_list[nb_name][(tnameA, tnameB)]=score
        self.sim_list=sim_list
        for nb_name in sim_list:
            if n_q_num > len(sim_list[nb_name]):
                nb_score[nb_name]=0
                continue
            sorted_list=sorted(sim_list[nb_name].items(), key=lambda d: d[1], reverse=True)
            visited_n_q=[]
            for i in range(min(n_q_num, len(sorted_list))):
                if sorted_list[i][0][0] in visited_n_q:
                    continue
                visited_n_q.append(sorted_list[i][0][0])
                nb_score[nb_name]+=sorted_list[i][1]
        return nb_score

    def existing_method_calc_table_sim_particular_nb(self, nb_name:str):
        # 2021/01/20修正
        #calc_var_start_time = timeit.default_timer()
        sim_list={}
        n_q_num=len(self.query_table)
        node_db_list=self.nb_node_list[nb_name]["Var"]
        for n_q in self.query_table:
            for n_db in node_db_list:
                if self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    continue
                sim=self.calc_rel_with_timecount(n_q, n_db)
                sim_list[(n_q, n_db)]=sim

        self.sim_list=sim_list
        ret_score=0.0
        
        sorted_list=sorted(sim_list.items(), key=lambda d: d[1], reverse=True)
        visited_n_q=[]
        for i in range(min(n_q_num, len(sim_list))):
            if sorted_list[i][0][0] in visited_n_q:
                continue
            visited_n_q.append(sorted_list[i][0][0])
            ret_score+=sorted_list[i][1]
        #calc_var_end_time = timeit.default_timer()
        #self.each_calc_time_sum["Var"]+=(calc_var_end_time-calc_var_start_time) #重複（calc_rel_with_timecountですでに足されている）
        return ret_score

    def existing_method_calc_code_sim(self):
        """
        Calculate code similarity for the code-based search method (C).
        セルを頭から結合したソースコードだけをみてtop-kのスコアにする．(self.nb_scoreにセット)
        """
        class_code_relatedness=self.class_code_relatedness

        # get query source code
        query_code=self.combining_query_code()


        for nb_name in self.valid_nb_name:
            # get db nodebook source code
            db_code_table=self.fetch_source_code_table(nb_name)
            db_code_list=db_code_table["cell_code"].values
            # ********* oldに対して追加点 ***********
            db_code_list2=[]
            for code in db_code_list:
                row_list=code.split("\n")
                for row in row_list:
                    if "#" in row:
                        row=row[:row.index("#")]
                        if row=="":
                            continue
                        db_code_list2.append(row)
                    else:
                        db_code_list2.append(row)
            if len(db_code_list2)==0:
                self.nb_score[nb_name]=0
                continue
            # ********* 上記，追加点 ***********
            db_code="\n".join(db_code_list2) # oldに対して変更点 db_code_list -> db_code_list2
            while "\n\n" in db_code:
                db_code=db_code.replace("\n\n", "\n")
            calc_code_start_time = timeit.default_timer()
            if self.calc_code_sim_method=="jaccard":
                sim=self.class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
            elif self.calc_code_sim_method=="levenshtein":
                sim=CodeSimilarity.code_sim_based_on_levenshtein_dist(query_code, db_code)
            else:
                sim=self.class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
            #sim=class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
            calc_code_end_time = timeit.default_timer()
            self.each_calc_time_sum["Cell"]+=calc_code_end_time-calc_code_start_time
            self.nb_score[nb_name]=sim


    def existing_method_calc_code_sim2(self):
        """
        1と2は結果を戻り値にするかself.nb_scoreに格納するかの違い．
        """
        nb_score={}
        # get query source code
        query_code=self.combining_query_code()


        for nb_name in self.valid_nb_name:
            # get db nodebook source code
            db_code_table=self.fetch_source_code_table(nb_name)
            db_code_list=db_code_table["cell_code"].values
            # ********* oldに対して追加点 ***********
            db_code_list2=[]
            for code in db_code_list:
                row_list=code.split("\n")
                for row in row_list:
                    if "#" in row:
                        row=row[:row.index("#")]
                        if row=="":
                            continue
                        db_code_list2.append(row)
                    else:
                        db_code_list2.append(row)
            if len(db_code_list2)==0:
                nb_score[nb_name]=0
                continue
            # ********* 上記，追加点 ***********
            db_code="\n".join(db_code_list2) # oldに対して変更点 db_code_list -> db_code_list2
            while "\n\n" in db_code:
                db_code=db_code.replace("\n\n", "\n")
            calc_code_start_time = timeit.default_timer()
            if self.calc_code_sim_method=="jaccard":
                sim=self.class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
            elif self.calc_code_sim_method=="levenshtein":
                sim=CodeSimilarity.code_sim_based_on_levenshtein_dist(query_code, db_code)
            else:
                sim=self.class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
            #sim=class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
            calc_code_end_time = timeit.default_timer()
            self.each_calc_time_sum["Cell"]+=calc_code_end_time-calc_code_start_time
            nb_score[nb_name]=sim
        return nb_score

            
    def set_based_method_calc_code_sim(self):
        """
        Calculate code similarity for the code-based search method (C).
        セルごとのソースコードをみてtop-kのスコアにする．(self.nb_scoreにセット)
        """
        class_code_relatedness=self.class_code_relatedness
        nb_score={}
        matched_pair=[]
        matched_pair2=[]

        # get query source code
        query_code=self.combining_query_code()

        for nb_name in self.valid_nb_name:
            sim_list=[]
            # get db nodebook source code
            db_code_table=self.fetch_source_code_table(nb_name)
            db_code_list=db_code_table["cell_code"].values
            # ********* oldに対して追加点 ***********
            for qNode in self.query_cell_code:
                query_code = self.query_cell_code[qNode]
                db_code_list2=[]
                code_id = 0
                for code in db_code_list:
                    code_id+=1
                    row_list=code.split("\n")
                    for row in row_list:
                        if "#" in row:
                            row=row[:row.index("#")]
                            if row=="":
                                continue
                            db_code_list2.append(row)
                        else:
                            db_code_list2.append(row)
                    if len(db_code_list2)==0:
                        sim_list.append((str(qNode), str(code_id), 0))
                        continue
                    db_code = "\n".join(db_code_list2)
                    while "\n\n" in db_code:
                        db_code=db_code.replace("\n\n", "\n")
                    calc_code_start_time = timeit.default_timer()
                    sim_tmp = class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
                    sim_list.append((str(qNode), str(code_id), sim_tmp))
                    calc_code_end_time = timeit.default_timer()
                    self.each_calc_time_sum["Cell"]+=calc_code_end_time-calc_code_start_time
                    db_code_list2=[]
            num=len(self.query_cell_code)
            self.sim_list_debug = sim_list
            sorted_sim_list = sorted(sim_list, key=lambda x: x[2], reverse=True) # 類似度が高い順にソート
            sim = 0
            i = 0
            j=0
            while(j<num):
                if i>=len(sorted_sim_list):
                    break
                if sorted_sim_list[i][0] not in matched_pair and sorted_sim_list[i][1] not in matched_pair2:
                    sim += sorted_sim_list[i][2]
                    matched_pair.append(sorted_sim_list[i][0])
                    matched_pair2.append(sorted_sim_list[i][1])
                    j+=1
                i+=1
            nb_score[nb_name]=sim
            
        return nb_score

    def combining_query_code(self) -> str:
        # get query source code
        stack_n=[self.query_root]
        query_code_list=[]
        while stack_n != []:
            node=stack_n.pop()
            if self.attr_of_q_node_type[node] == "Cell":
                query_code_list.append(self.query_cell_code[node])
            stack_n.extend(list(self.QueryGraph.successors(node)))
        query_code="\n".join(query_code_list)
        while "\n\n" in query_code:
            query_code=query_code.replace("\n\n", "\n")
        return query_code
            
    def existing_method_calc_code_sim_particular_nb(self, nb_name, query_code):
        """
        セルを頭から結合したソースコードだけをみてtop-kのスコアにする．(self.nb_scoreにセット)
        """
        class_code_relatedness=self.class_code_relatedness

        # get db nodebook source code
        db_code_table=self.fetch_source_code_table(nb_name)
        db_code_list=db_code_table["cell_code"].values
        # ********* oldに対して追加点 ***********
        db_code_list2=[]
        for code in db_code_list:
            row_list=code.split("\n")
            for row in row_list:
                if "#" in row:
                    row=row[:row.index("#")]
                    if row=="":
                        continue
                    db_code_list2.append(row)
                else:
                    db_code_list2.append(row)
        if len(db_code_list2)==0:
            #self.nb_score[nb_name]=0
            return 0
        db_code="\n".join(db_code_list2) # oldに対して変更点 db_code_list -> db_code_list2
        while "\n\n" in db_code:
            db_code=db_code.replace("\n\n", "\n")
        calc_code_start_time = timeit.default_timer()
        sim=class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
        calc_code_end_time = timeit.default_timer()
        self.each_calc_time_sum["Cell"]+=calc_code_end_time-calc_code_start_time
        return sim
            

    def existing_method_sum(self, if_codeSimPerCell=False):
        """
        Set-based search method w/o optimization.
        Save computational notebook similarity in self.nb_score.
        """
        if if_codeSimPerCell:
            self.init_arranged_all_weights()
        else:
            self.init_arranged_dataandoutput_weights()

        self.init_each_calc_time_sum()
        self.nb_score={}
        nb_score_according_to_table_sim={}
        if self.w_v!=0:
            self.existing_method_calc_table_sim()
            nb_score_according_to_table_sim=self.nb_score
            self.nb_score={}
        nb_score_according_to_code_sim={}
        if self.w_c!=0:
            if if_codeSimPerCell:
                # セル単位の類似度
                nb_score_according_to_code_sim = self.set_based_method_calc_code_sim()
            else:
                self.existing_method_calc_code_sim()
                nb_score_according_to_code_sim=self.nb_score
            nb_score_according_to_code_sim=self.nb_score
            self.nb_score={}

        for nb_name in self.valid_nb_name:
            if nb_name not in nb_score_according_to_table_sim:
                nb_score_according_to_table_sim[nb_name]=0
            if nb_name not in nb_score_according_to_code_sim:
                nb_score_according_to_code_sim[nb_name]=0
            if self.w_l!=0:
                lib_list2=self.fetch_all_library_from_db(nb_name)
                lib_rel=self.calc_lib_score(self.query_lib, lib_list2)
            else:
                lib_rel=0
            if self.w_d!=0:
                output_rel = self.previous_calc_output_sim(nb_name)
            else:
                output_rel=0
            self.nb_score[nb_name]=lib_rel * self.w_l + nb_score_according_to_table_sim[nb_name] * self.w_v + nb_score_according_to_code_sim[nb_name] * self.w_c + output_rel*self.w_d

    def existing_method_sum_fast_method(self, k:int, if_codeSimPerCell=False, flg_juneau=False): 
        """
        Set-based search method with optimization.
        Save computational notebook similarity in self.nb_score.

        Args:
            k (int): a natural number for top-k search on computational notebooks.
        """
        if if_codeSimPerCell:
            self.init_arranged_all_weights()
        else:
            self.init_arranged_dataandoutput_weights()
            
        self.k=k
        self.init_each_calc_time_sum()
        self.nb_score={}
        
        if flg_juneau:
            self.init_query_col()
            self.set_all_querytable_col()

        table_q_num=len(self.query_table)
        code_q_num=len(self.query_cell_code)
        if not self.flg_running_faster["flg_prune_under_sim"]:
            self.existing_method_sum(if_codeSimPerCell=if_codeSimPerCell)
        else:
            if self.w_v==0:
                self.existing_method_sum_fast_method_w_v0()
            else:
                current_nb_score={}
                nb_score_according_to_code_sim={}
                if self.w_c!=0:
                    if if_codeSimPerCell:
                        # セル単位の類似度
                        nb_score_according_to_code_sim = self.set_based_method_calc_code_sim()
                    else:
                        self.existing_method_calc_code_sim()
                        nb_score_according_to_code_sim=self.nb_score
                    self.nb_score={}

                for nb_name in self.valid_nb_name:
                    if nb_name not in nb_score_according_to_code_sim:
                        nb_score_according_to_code_sim[nb_name]=0
                    if self.w_l!=0:
                        lib_list2=self.fetch_all_library_from_db(nb_name)
                        lib_rel=self.calc_lib_score(self.query_lib, lib_list2)
                    else:
                        lib_rel=0
                    current_nb_score[nb_name] = lib_rel * self.w_l
                    if code_q_num!=0:
                        current_nb_score[nb_name] += nb_score_according_to_code_sim[nb_name] * self.w_c / code_q_num
                    if self.w_d!=0:
                        current_nb_score[nb_name]+= self.previous_calc_output_sim(nb_name)*self.w_d

                n_q_num=len(self.query_table)
                if self.flg_running_faster["flg_optimize_calc_order"]:
                    sorted_nb_score=sorted(current_nb_score.items(), key=lambda d: d[1], reverse=True)
                else:
                    sorted_nb_score=list(current_nb_score.items())
                #logging.info(f"self.each_calc_time_sum[Var]: {self.each_calc_time_sum["Var"]}")
                if self.w_v!=0:
                    for i in range(len(sorted_nb_score)):
                        nb_name=sorted_nb_score[i][0]
                        #current_score=sorted_nb_score[i][1]
                        if len(self.nb_score)<self.k:
                            k_score=0
                        else:
                            self.top_k_score=sorted(self.nb_score.items(), key=lambda d: d[1], reverse=True)
                            k_score=self.top_k_score[self.k-1][1]
                        if table_q_num!=0:
                            if self.flg_prune_under_sim_for_new_proposal_method(v_count=n_q_num, current_score=current_nb_score[nb_name], compare_score=k_score, max_v_score=self.w_v/table_q_num):
                                continue
                            self.nb_score[nb_name]=current_nb_score[nb_name]+self.w_v*self.existing_method_calc_table_sim_particular_nb(nb_name)/table_q_num
            
    def existing_method_sum_fast_method_w_v0(self):
        """
        Call this function for set-based search method with optimization if data weight is zero.
        Save computational notebook similarity in self.nb_score.
        """
        current_nb_score={}
        for nb_name in self.valid_nb_name:
            if self.w_l!=0:
                lib_list2=self.fetch_all_library_from_db(nb_name)
                current_nb_score[nb_name]=self.calc_lib_score(self.query_lib, lib_list2) * self.w_l
            else:
                current_nb_score[nb_name]=0
            if self.w_d!=0:
                current_nb_score[nb_name]+= self.previous_calc_output_sim(nb_name)*self.w_d
        if self.w_c!=0:
            query_code=self.combining_query_code()
            #n_q_num=len(self.query_cell_code)
            sorted_nb_score=sorted(current_nb_score.items(), key=lambda d: d[1], reverse=True)
            for i in range(len(sorted_nb_score)):
                nb_name=sorted_nb_score[i][0]
                if len(self.nb_score)<self.k:
                    k_score=0
                else:
                    self.top_k_score=sorted(self.nb_score.items(), key=lambda d: d[1], reverse=True)
                    k_score=self.top_k_score[self.k-1][1]
                self.nb_score[nb_name]=current_nb_score[nb_name]
                if self.flg_prune_under_sim_for_new_proposal_method_3(self.w_c, remain_count=1, current_score=current_nb_score[nb_name], compare_score=k_score, max_score=self.w_c):
                    continue
                self.nb_score[nb_name]+=self.w_c*self.existing_method_calc_code_sim_particular_nb(nb_name, query_code)
                #print(self.nb_score[nb_name])
        else:
            self.nb_score=current_nb_score
            
    def get_table_size1(self, table:pd.DataFrame):
        """
        For development.
        テーブルのカラム数および行数を調べる．

        Args:
            table (DataFrame)
        
        Returns:
            tuple(int-like, int-like): num of columns and index
        """
        if type(table) != type(pd.DataFrame()):
            logging.info(f"error: not DataFrame object")
            return 0, 0

        return len(table.columns), len(table.index)

    def get_table_size2(self, table):
        """
        For development.
        テーブルのカラム数，行数に加えて，
        データ型ごとのカラム数を調べる．

        Args:
            table (DataFrame)

        Returns:
            triple(int-like, int-like, dict{str: int-like}): 
                num of columns and index, and dict. 
                dict has key that is column type and value that is num of columns of the type.
        """
        num_of_columns={}

        if type(table) != type(pd.DataFrame()):
            logging.info(f"error: not DataFrame object")
            return 0, 0, num_of_columns

        for col_name in table.columns:
            col_type = table[col_name].dtype
            if col_type not in num_of_columns:
                num_of_columns[col_type]=0
            num_of_columns[col_type]+=1

        return len(table.columns), len(table.index), num_of_columns
    
    def get_table_size3(self, table, kilo=False):
        """
        For development.
        テーブルサイズをバイト数で返す．

        Args:
            table (DataFrame)
        
        Returns:
            int-like: table size Bytes
        """
        # 参考: http://www.mwsoft.jp/programming/numpy/dataframe_memory.html
        if type(table) != type(pd.DataFrame()):
            logging.info(f"error: not DataFrame object")
            return 0

        df_columns_size=table.memory_usage(deep=True)
        df_size_sum=0
        for size in df_columns_size:
            df_size_sum+=size

        if kilo:
            return df_size_sum/1024
        return df_size_sum



    # Juneauの適用

    def sketch_query_cols(self, query:pd.DataFrame, sz:int=5) -> list:# -> List[str]:
        # 疑問: データがfloat型のみの場合はサンプリングがうまくいかない(szより列数が小さくなる可能性がある)と思われるが, どうなのか.(sketch_column_and_row_for_meta_mappingも)
        """
        引数のテーブルqueryに対し，引数szで指定する最大列数以下のテーブルの列名をlistで返す．
        最大列数まで列数を減らすときは、その列の値ができるだけバラバラな値をとる順(|セット集合|/|多重集合|が大きい順)に残す．

        Args:
            query (DataFrame): クエリ(テーブル).
            sz (int): 列数の最大サイズ．defaults to 5.

        Returns:
            list [str]: 指定の最大列数だけサンプリングしたテーブルの列名.元の列数が指定の最大列数より少ない場合は，全ての列名のリスト．
        """
        if query.shape[1] <= sz: # queryの列数がsz以下の時. (query.shape[1]: queryの列数)
            return query.columns.tolist() # .tolist(): リスト型listに変換
        else: # queryの列数がszより多い場合はサンプリングする.
            q_cols = query.columns.tolist() # q_cols (list[str]): queryの列名リスト.
            c_scores = []
            for i in q_cols: # 各列に対して以下の操作
                if i == "Unnamed: 0" or "index" in i:
                    continue
                if query[i].dtype is np.dtype(float): # テーブルqueryの列名iのデータ型がfloatのときcontinue
                    continue
                cs_v = query[i].tolist()
                # |セット集合|/|多重集合|を計算
                c_scores.append((i, float(len(set(cs_v))) / float(len(cs_v))))
            # その列の値ができるだけバラバラな値をとる順にソートする．
            c_scores = sorted(c_scores, key=lambda d: d[1], reverse=True) #降順ソート

            q_cols_chosen = []
            c_count = 0
            for i, j in c_scores: # i (str): 列名
                if c_count == sz:
                    break
                q_cols_chosen.append(i)
                c_count += 1
            return q_cols_chosen


    def sketch_column_and_row_for_meta_mapping(self, sz:int=5, row_size:int=1000):
        # 疑問: データがfloat型のみの場合はサンプリングがうまくいかない(szより列数が小さくなる可能性がある)と思われるが, どうなのか.(sketch_query_colsも)
        """
        テーブルself.schema_elementについて，各テーブルグループに対して列数および行数が指定の最大数以下となるように，
        テーブルグループごとにデータをサンプリングする．
        サンプリングの結果はself.schema_element_sample_colに格納．

        Args:
            sz (int): サンプリングする最大の列(col)数．defaults to 5.
            row_size (int): サンプリングする最大の行(row)数．defaults to 1000. 

        self.var str:
            self.schema_element (dict{int: dict{str: list[]}):
                テーブルグループごとの，各列名の列の実際のデータ値の重複無し集合．(null値を除く.)
                {テーブルグループID: {列名: [同じ列名を持つグループ内のすべてのテーブルから集めた実際のデータ値]}}
            self.schema_element_sample_col (dict{int: dict{str: list[]}):
                self.schema_elementの各テーブルグループに対して指定のサイズ以下の列数，行数となるようにサンプリングしたもの．
        """
        self.schema_element_sample_col = {}
        for i in self.schema_element.keys(): # 各テーブルグループに対し以下の処理
            self.schema_element_sample_col[i] = {}
            if len(self.schema_element[i].keys()) <= sz: # テーブルグループの列数がsz個以下の時
                # 表データの行のサンプリング
                # scはschemaの略か.
                for sc in self.schema_element[i].keys(): # 指定テーブルグループの各列に対して
                    if len(self.schema_element[i][sc]) < row_size: # 最大行数よりもサンプリング対象の行数が少ない場合は, すべての行を得る.
                        self.schema_element_sample_col[i][sc] = self.schema_element[i][sc]
                    else: # 最大行数よりもサンプリング対象の行数が多い場合は, 最大行数分だけをランダムに選んで抽出する.
                        self.schema_element_sample_col[i][sc] = random.sample(
                            self.schema_element[i][sc], row_size
                        )
            else: # サンプリング対象の列数がsz列より多いとき
                ##
                # sc_choice ([str, float]): strは列名, floatはその列のデータ(値)の集合に対して, 
                # |セット集合|/|多重集合|を計算している.(セット集合 = 重複無し集合). 
                # scはschemaの略か.
                ##
                sc_choice = []
                for sc in self.schema_element[i].keys():
                    if sc == "Unnamed: 0" or "index" in sc:
                        continue
                    if self.schema_element_dtype[i][sc] is np.dtype(float):
                        continue
                    sc_value = list(self.schema_element[i][sc]) # テーブルグループIDがi, 列名がscの実データ値をsc_valueに格納
                    # sc列目のデータ集合に対し, |セット集合|/|多重集合|を計算
                    sc_choice.append(
                        (sc, float(len(set(sc_value))) / float(len(sc_value)))
                    )
                # その列の値ができるだけバラバラな値をとる順にソートする．
                sc_choice = sorted(sc_choice, key=lambda d: d[1], reverse=True) #降順

                count = 0
                for sc, v in sc_choice:
                    if count == sz:
                        break
                    # 表データの行のサンプリング
                    if len(self.schema_element[i][sc]) < row_size: # 最大行数よりもサンプリング対象の表データの行数が少ない場合は, すべての行を得る.
                        self.schema_element_sample_col[i][sc] = self.schema_element[i][
                            sc
                        ]
                    else: # 最大行数よりもサンプリング対象の表データの行数が多い場合は, 最大行数分だけをランダムに選んで抽出する.
                        self.schema_element_sample_col[i][sc] = random.sample(
                            self.schema_element[i][sc], row_size
                        )

                    count += 1


    def init_schema_mapping(self, max_table_groups:int=1001001):
        """
        Original: Juneau/search_withprov.py

        開発用．        
        """

        logging.info("Start Reading From Neo4j!")
        start_time=timeit.default_timer()
        #matcher = NodeMatcher(self.geng)
        #matcher = NodeMatcher(graph_db)

        #tables_touched = [] # list[str]: tables_connectedにすでに入れたテーブル名のリスト.
        tables_connected = [] # list[list[str]]: グループ(ワークフローグラフで接続された関係にあるもののグループ)ごとのテーブル名リストのリスト．
        # self.real_tables ({str: DataFrame}): (テーブル名:実際のテーブルの内容)の辞書. 
        # テーブル名はf"rtable{idi}"またはf"rtable{idi}_{vid}"となっている． 
        # idiはintまたはstr(変数名)か. vidはversion IDのことか.
        #for i in self.real_tables.keys(): # i (str): テーブル名
        #    # テーブルとその従属関係があるテープルをすべて探す.
        #    if i[6:] not in set(tables_touched): # i[6:]: テーブル名から"rtables"を除いた部分.変数名か.
        #        logging.info(i)
        #        current_node = matcher.match("Var", name=i[6:]).first() # current_node (Node): 変数のノード
        #        connected_tables = self.dfs(current_node) # connected_tables (str): 変数 idi に従属関係のあるテーブルの変数名リスト.
        #        tables_touched = tables_touched + connected_tables # リストの連結
        #        tables_connected.append(connected_tables)
        self.real_tables={}
        for nb_nameA in self.valid_nb_name:
            connected_tables = []
            for node, nb_nameB in self.attr_of_db_nb_name.items():
                if nb_nameA != nb_nameB:
                    continue
                if self.attr_of_db_node_type[node] != "Var":
                    continue
                
                real_table=self.fetch_var_table(node)
                if "Unnamed: 0" in real_table.columns: #列名に不備がある場合はその列を削除
                    real_table.drop(["Unnamed: 0"], axis=1, inplace=True)
                if "Unnamed:0" in real_table.columns: #列名に不備がある場合はその列を削除
                    real_table.drop(["Unnamed:0"], axis=1, inplace=True)
                self.real_tables[node]=real_table

                connected_tables.append(node)
            tables_connected.append(connected_tables)

            if len(tables_connected) > max_table_groups: #テスト動作用
                break
                
            
        self.schema_linking = {}
        #schema_linking = {}
        self.schema_element = {}
        #schema_element = {}
        self.schema_element_count = {}
        #schema_element_count = {}
        self.schema_element_dtype = {}
        #schema_element_dtype = {}

        self.table_group = {}
        #table_group = {}

        # assign each table a group id
        for idi, i in enumerate(tables_connected):
            for j in i:
                self.table_group[j] = idi # self.table_group[テーブル名(変数名または変数名_vid?)]=グループID
                #table_group[j] = idi # self.table_group[テーブル名(変数名または変数名_vid?)]=グループID

        for idi, i in enumerate(tables_connected):
            # idi (int): enumerateによる, ループごとにインクリメントする整数.
            # i (str): 変数名または変数名_vid. "rtable"+iでテーブル名になる.
            self.schema_linking[idi] = {}
            #schema_linking[idi] = {}
            self.schema_element[idi] = {}
            #schema_element[idi] = {}
            self.schema_element_dtype[idi] = {}
            #schema_element_dtype[idi] = {}
            self.schema_element_count[idi] = {}
            #schema_element_count[idi] = {}

            for j in i: # 各テーブルに対し
                #tname = "rtable" + j
                tname = j
                #if tname not in self.real_tables: # テーブル名エラーの時
                #    continue
                
                #for col in self.real_tables[tname].columns: # テーブルの列それぞれに対して以下の操作
                real_table = self.real_tables[tname] # 代替方法として追加. 以下，"real_tables[tname]"をreal_tableに変更済み．
                #real_tables[tname]=real_table
                    
                for col in real_table.columns: # テーブルの列それぞれに対して以下の操作
                    # テーブルの列名ごとに固有のsidを付与.
                    if col not in self.schema_linking[idi]:
                        if len(self.schema_linking[idi].keys()) == 0: # まだself.schema_linking[idi]に要素が入っていないとき(ループ1周目)
                            sid = 0
                        else:
                            sid = max(list(self.schema_linking[idi].values())) + 1

                        self.schema_linking[idi][col] = sid
                        self.schema_element_dtype[idi][col] = real_table[
                            col
                        ].dtype
                        self.schema_element_count[idi][col] = 1
                        self.schema_element[idi][col] = []
                        #schema_element[idi][col] += real_tables[tname][col][
                        self.schema_element[idi][col] += real_table[col][
                            real_table[col].notnull() # Null値でないときにTrue
                        ].tolist()
                        self.schema_element[idi][col] = list(
                            set(self.schema_element[idi][col])
                        )
                    else:
                        self.schema_element[idi][col] += real_table[col][
                            real_table[col].notnull()
                        ].tolist()
                        self.schema_element[idi][col] = list(
                            set(self.schema_element[idi][col])
                        )
                        self.schema_element_count[idi][col] += 1
        end_time=timeit.default_timer()
        logging.info("There are %s groups of tables." % len(tables_connected))
        logging.info(f"Initialized schema mapping: {end_time-start_time} Seconds.")
        #return tables_connected, schema_linking, schema_element, schema_element_count, schema_element_dtype, table_group

    def save_schema_mapping_json(self):
        """
        開発実験用．
        """
        current_dir_path = os.getcwd()

        save_tmp={}
        for i in self.schema_element.keys():
            try:
                save_tmp[i]=json.dumps(self.schema_element[i])
            except:
                pass
        with open(f"{current_dir_path}/sample_profiled_dataset/schema_element.json", 'w', encoding='utf-8') as f:
            json.dump(save_tmp, f, indent=2, ensure_ascii=False)
        with open(f"{current_dir_path}/sample_profiled_dataset/schema_linking.json", 'w', encoding='utf-8') as f:
            json.dump(self.schema_linking, f, indent=2, ensure_ascii=False)
        with open(f"{current_dir_path}/sample_profiled_dataset/schema_element_count.json", 'w', encoding='utf-8') as f:
            json.dump(self.schema_element_count, f, indent=2, ensure_ascii=False)
        #with open(f"{current_dir_path}/sample_profiled_dataset/schema_element_dtype.json", 'w', encoding='utf-8') as f:
        #    json.dump(self.schema_element_dtype, f, indent=2, ensure_ascii=False)
        with open(f"{current_dir_path}/sample_profiled_dataset/table_group.json", 'w', encoding='utf-8') as f:
            json.dump(self.table_group, f, indent=2, ensure_ascii=False)
        
    def init_schema_mapping_json(self):
        """
        開発実験用．
        """
        current_dir_path = os.getcwd()
        with open(f"{current_dir_path}/sample_profiled_dataset/schema_element.json", encoding='utf-8') as f:
            self.schema_element = json.load(f)    
        with open(f"{current_dir_path}/sample_profiled_dataset/schema_linking.json", encoding='utf-8') as f:
            self.schema_linking = json.load(f)
        with open(f"{current_dir_path}/sample_profiled_dataset/schema_element_count.json", encoding='utf-8') as f:
            self.schema_element_count = json.load(f)
        #with open(f"{current_dir_path}/sample_profiled_dataset/schema_element_dtype.json", encoding='utf-8') as f:
        #    self.schema_element_dtype = json.load(f)
        with open(f"{current_dir_path}/sample_profiled_dataset/table_group.json", encoding='utf-8') as f:
            self.table_group = json.load(f)


    def init_schema_mapping2(self, max_table_groups=1001001):
        """
        開発用
        """
        # search_withprov.pyから

        logging.info("Start Reading From Neo4j!")
        tables_connected = []
        self.real_tables={}
        for nb_nameA in self.valid_nb_name:
            connected_tables = []
            for node, nb_nameB in self.attr_of_db_nb_name.items():
                if nb_nameA != nb_nameB:
                    continue
                if self.attr_of_db_node_type[node] != "Var":
                    continue

                real_table=self.fetch_var_table(node)
                if "Unnamed: 0" in real_table.columns: #列名に不備がある場合はその列を削除
                    real_table.drop(["Unnamed: 0"], axis=1, inplace=True)
                if "Unnamed:0" in real_table.columns: #列名に不備がある場合はその列を削除
                    real_table.drop(["Unnamed:0"], axis=1, inplace=True)
                self.real_tables[node]=real_table

                connected_tables.append(node)
            tables_connected.append(connected_tables)


            if len(tables_connected) > max_table_groups: #テスト動作用
                break
                
            
        self.schema_linking = {}
        self.schema_element = {}
        self.schema_element_count = {}
        self.schema_element_dtype = {}
        self.table_group = {}
        self.init_schema_mapping_json()


        for idi, i in enumerate(tables_connected):
            flg_schema_element=False
            if idi not in self.schema_element:
                self.schema_element[idi] = {}
                flg_schema_element=True
            self.schema_element_dtype[idi] = {}
            for j in i: # 各テーブルに対し
                #tname = "rtable" + j
                tname = j
                #if tname not in self.real_tables: # テーブル名エラーの時
                #    continue
                
                #for col in self.real_tables[tname].columns: # テーブルの列それぞれに対して以下の操作
                real_table = self.real_tables[tname] # 代替方法として追加. 以下，"real_tables[tname]"をreal_tableに変更済み．
                #real_tables[tname]=real_table
                    
                if flg_schema_element:
                    for col in real_table.columns: # テーブルの列それぞれに対して以下の操作
                        # テーブルの列名ごとに固有のsidを付与.
                        if col not in self.schema_linking[idi]:
                            self.schema_element_dtype[idi][col] = real_table[
                                col
                            ].dtype
                            self.schema_element[idi][col] = []
                            #schema_element[idi][col] += real_tables[tname][col][
                            self.schema_element[idi][col] += real_table[col][
                                real_table[col].notnull() # Null値でないときにTrue
                            ].tolist()
                            self.schema_element[idi][col] = list(
                                set(self.schema_element[idi][col])
                            )
                        else:
                            self.schema_element[idi][col] += real_table[col][
                                real_table[col].notnull()
                            ].tolist()
                            self.schema_element[idi][col] = list(
                                set(self.schema_element[idi][col])
                            )
                else:
                    for col in real_table.columns: # テーブルの列それぞれに対して以下の操作
                        # テーブルの列名ごとに固有のsidを付与.
                        if col not in self.schema_linking[idi]:
                            self.schema_element_dtype[idi][col] = real_table[
                                col
                            ].dtype
                        else:
                            pass

        logging.info("There are %s groups of tables." % len(tables_connected))
        #return tables_connected, schema_linking, schema_element, schema_element_count, schema_element_dtype, table_group

    def init_schema_mapping3(self, max_table_groups=1001001):
        """
        開発用
        """
        # search_withprov.pyから

        logging.info("Start Reading From Neo4j!")
        tables_connected = []
        self.real_tables={}
        for nb_nameA in self.valid_nb_name:
            connected_tables = []
            for node, nb_nameB in self.attr_of_db_nb_name.items():
                if nb_nameA != nb_nameB:
                    continue
                if self.attr_of_db_node_type[node] != "Var":
                    continue

                real_table=self.fetch_var_table(node)
                if "Unnamed: 0" in real_table.columns: #列名に不備がある場合はその列を削除
                    real_table.drop(["Unnamed: 0"], axis=1, inplace=True)
                if "Unnamed:0" in real_table.columns: #列名に不備がある場合はその列を削除
                    real_table.drop(["Unnamed:0"], axis=1, inplace=True)
                self.real_tables[node]=real_table

                connected_tables.append(node)
            tables_connected.append(connected_tables)


            if len(tables_connected) > max_table_groups: #テスト動作用
                break
                

        logging.info("There are %s groups of tables." % len(tables_connected))
        #return tables_connected, schema_linking, schema_element, schema_element_count, schema_element_dtype, table_group



    """
    def init_schema_mapping2(self, max_table_groups=1001001, thres=0.7):
        # search_withprov.pyから

        logging.info("Start Reading From Neo4j!")
        #matcher = NodeMatcher(self.geng)
        #matcher = NodeMatcher(graph_db)

        #tables_touched = [] # list[str]: tables_connectedにすでに入れたテーブル名のリスト.
        tables_connected = [] # list[list[str]]: グループ(ワークフローグラフで接続された関係にあるもののグループ)ごとのテーブル名リストのリスト．
        # self.real_tables ({str: DataFrame}): (テーブル名:実際のテーブルの内容)の辞書. 
        # テーブル名はf"rtable{idi}"またはf"rtable{idi}_{vid}"となっている． 
        # idiはintまたはstr(変数名)か. vidはversion IDのことか.
        #for i in self.real_tables.keys(): # i (str): テーブル名
        #    # テーブルとその従属関係があるテープルをすべて探す.
        #    if i[6:] not in set(tables_touched): # i[6:]: テーブル名から"rtables"を除いた部分.変数名か.
        #        logging.info(i)
        #        current_node = matcher.match("Var", name=i[6:]).first() # current_node (Node): 変数のノード
        #        connected_tables = self.dfs(current_node) # connected_tables (str): 変数 idi に従属関係のあるテーブルの変数名リスト.
        #        tables_touched = tables_touched + connected_tables # リストの連結
        #        tables_connected.append(connected_tables)
        self.real_tables={}
        for nb_nameA in self.valid_nb_name:
            for node, nb_nameB in self.attr_of_db_nb_name.items():
                if nb_nameA != nb_nameB:
                    continue
                if self.attr_of_db_node_type[node] != "Var":
                    continue
                self.real_tables[node]=self.fetch_var_table(node)
            if len(self.real_tables) > max_table_groups: #テスト動作用
                break
                
            
        self.schema_linking = {}
        #schema_linking = {}
        self.schema_element = {}
        #schema_element = {}
        self.schema_element_count = {}
        #schema_element_count = {}
        self.schema_element_dtype = {}
        #schema_element_dtype = {}

        self.table_group = {}
        #table_group = {}

        # assign each table a group id
        #for idi, i in enumerate(tables_connected):
        #    for j in i:
        #        self.table_group[j] = idi # self.table_group[テーブル名(変数名または変数名_vid?)]=グループID
                #table_group[j] = idi # self.table_group[テーブル名(変数名または変数名_vid?)]=グループID

        idi=-1
        score=0
        for tname in self.real_tables:
            if score < thres:
                idi += 1
                # idi (int): enumerateによる, ループごとにインクリメントする整数.
                # i (str): 変数名または変数名_vid. "rtable"+iでテーブル名になる.
                self.schema_linking[idi] = {}
                #schema_linking[idi] = {}
                self.schema_element[idi] = {}
                #schema_element[idi] = {}
                self.schema_element_dtype[idi] = {}
                #schema_element_dtype[idi] = {}
                self.schema_element_count[idi] = {}
                #schema_element_count[idi] = {}

            for j in i: # 各テーブルに対し
                #tname = "rtable" + j
                tname = j
                #if tname not in self.real_tables: # テーブル名エラーの時
                #    continue
                
                #for col in self.real_tables[tname].columns: # テーブルの列それぞれに対して以下の操作
                real_table = self.real_tables[tname]
                if "Unnamed: 0" in real_table.columns: #列名に不備がある場合はその列を削除
                    real_table.drop(["Unnamed: 0"], axis=1, inplace=True)
                if "Unnamed:0" in real_table.columns: #列名に不備がある場合はその列を削除
                    real_table.drop(["Unnamed:0"], axis=1, inplace=True)
                #real_tables[tname]=real_table
                    
                for col in real_table.columns: # テーブルの列それぞれに対して以下の操作
                    # テーブルの列名ごとに固有のsidを付与.
                    if col not in self.schema_linking[idi]:
                        if len(self.schema_linking[idi].keys()) == 0: # まだself.schema_linking[idi]に要素が入っていないとき(ループ1周目)
                            sid = 0
                        else:
                            sid = max(list(self.schema_linking[idi].values())) + 1

                        self.schema_linking[idi][col] = sid
                        self.schema_element_dtype[idi][col] = real_table[
                            col
                        ].dtype
                        self.schema_element_count[idi][col] = 1
                        self.schema_element[idi][col] = []
                        #schema_element[idi][col] += real_tables[tname][col][
                        self.schema_element[idi][col] += real_table[col][
                            real_table[col].notnull() # Null値でないときにTrue
                        ].tolist()
                        self.schema_element[idi][col] = list(
                            set(self.schema_element[idi][col])
                        )
                    else:
                        self.schema_element[idi][col] += real_table[col][
                            real_table[col].notnull()
                        ].tolist()
                        self.schema_element[idi][col] = list(
                            set(self.schema_element[idi][col])
                        )
                        self.schema_element_count[idi][col] += 1
        logging.info("There are %s groups of tables." % len(tables_connected))
        #return tables_connected, schema_linking, schema_element, schema_element_count, schema_element_dtype, table_group

    """

    def sample_rows_for_each_column(self, row_size:int=1000):
        """
        テーブルself.schema_elementについて，各テーブルグループに対して行数が指定の最大行数以下となるように，
        テーブルグループごとにデータをサンプリングする．
        サンプリングの結果はself.schema_element_sample_rowに格納．

        Args:
            row_size (int): サンプリングする最大の行数．defaults to 1000. 

        self.var str:
            self.schema_element ({int: {str: list[]}):
                テーブルグループごとの，各列名の列の実際のデータ値の重複無し集合．(null値を除く.)
                {テーブルグループID: {列名: [同じ列名を持つグループ内のすべてのテーブルから集めた実際のデータ値]}}
            self.schema_element_sample_row ({int/str: {str: list[]}}):
                self.schema_elementの各テーブルグループに対して指定のサイズ以下の行数となるようにサンプリングしたもの．
        """
        self.schema_element_sample_row = {}
        for i in self.schema_element.keys(): # 各テーブルに対して（i:テーブル名）
            self.schema_element_sample_row[i] = {}
            for sc in self.schema_element[i].keys(): #各列に対して(sc:列名)
                if len(self.schema_element[i][sc]) < row_size: # サンプリング前のデータ数（行数）がrow_sizeより小さければそのままを保持
                    self.schema_element_sample_row[i][sc] = self.schema_element[i][sc]
                else: #サンプリング前のデータ数（行数）がrow_sizeより大きれば, 行をサンプリングして行数をrow_sizeにする．
                    self.schema_element_sample_row[i][sc] = random.sample(
                        self.schema_element[i][sc], row_size
                    )


    def schema_mapping(self, tableA:pd.DataFrame, tableB:pd.DataFrame, meta_mapping:dict, gid:int) -> Tuple[dict, float]:
        """
        Original: Juneau/withprov_opt.py

        スキーママッピング（2つの表の列の対応関係）を作成する．
        """
        s_mapping = {} # dict{}: 内容は{sid: 列名}
        t_mapping = {} # dict{}: 内容は{sid: 列名}
        for i in tableA.columns.tolist(): # tableAの列名のリスト. (.tolist(): PandasのDataFrame型をリスト型listに変換)
            # i (str): 列名
            if i not in meta_mapping[gid]:
                continue
            # meta_mappingのグループIDがgidに同名の列名が存在する場合
            t_mapping[self.schema_linking[gid][meta_mapping[gid][i]]] = i

        for i in tableB.columns.tolist(): # tableBの列名それぞれに対し
            # i (str): 列名
            if self.schema_linking[gid][i] in t_mapping: # tableAとtableBで共通のsidが存在する場合
                if ( # tableAとtableBで列名iの列のデータ型が異なる場合
                    tableB[i].dtype
                    != tableA[t_mapping[self.schema_linking[gid][i]]].dtype
                ):
                    continue
                # tableAとtableBで列名iの列のデータ型が同じ場合
                s_mapping[t_mapping[self.schema_linking[gid][i]]] = i

        max_valueL = []
        for i in s_mapping.keys():
            j = s_mapping[i] # j (str): 列名
            max_valueL.append(self.row_similarity(tableA[i], tableB[j]))

        if len(max_valueL) > 0:
            mv = max(max_valueL)
        else:
            mv = 0

        return s_mapping, mv

    def comp_table_similarity_key(
        self, 
        SM_test,
        tableA:pd.DataFrame,
        tableB:pd.DataFrame,
        SM:dict,
        gid:int,
        meta_mapping,
        schema_linking,
        thres_key_prune,
        thres_key_cache,
        unmatched,
    ):
        """
        withprov_optから
        """

        key_choice = []
        for kyA in SM.keys():
            flg = False
            if kyA in self.already_map[gid]:
                determined = self.already_map[gid][kyA]
                check_set = set(list(SM.values()))
                for ds in determined:
                    if ds.issubset(check_set):
                        flg = True
                        break
                if flg:
                    key_choice.append(
                        (
                            kyA,
                            self.app_common_key(
                                tableA, tableB, SM, kyA, thres_key_prune
                            ),
                        )
                    )
                    break

                else:
                    key_score = self.app_common_key(
                        tableA, tableB, SM, kyA, thres_key_prune
                    )
                    key_choice.append((kyA, key_score))
                    if key_score == 1:
                        break
            else:
                key_score = self.app_common_key(
                    tableA, tableB, SM, kyA, thres_key_prune
                )
                key_choice.append((kyA, key_score))

        if len(key_choice) == 0:
            return 0, meta_mapping, unmatched, 0, None
        else:
            key_choice = sorted(key_choice, key=lambda d: d[1], reverse=True)
            key_chosen = key_choice[0][0]
            key_factor = key_choice[0][1]

            if key_factor >= thres_key_cache:
                if key_chosen not in self.already_map[gid]:
                    self.already_map[gid][key_chosen] = []
                self.already_map[gid][key_chosen].append(set(list(SM.values())))

            (
                SM_real,
                meta_mapping,
                unmatched,
                sm_time,
            ) = SM_test.mapping_naive_incremental(
                tableA, tableB, gid, meta_mapping, schema_linking, unmatched, mapped=SM
            )

            row_sim = self.row_similarity(
                tableA[key_chosen], tableB[SM_real[key_chosen]]
            )
            col_sim = self.col_similarity(tableA, tableB, SM_real, key_factor)

            return col_sim, row_sim, meta_mapping, unmatched, sm_time, key_chosen


    def app_common_key(self, tableA:pd.DataFrame, tableB:pd.DataFrame, SM:dict, key, thres_prune:float) -> float:
        kyA = key
        kyB = SM[key]
        key_value_A = tableA[kyA].tolist()
        key_value_B = tableB[kyB].tolist()

        key_estimateA = float(len(set(key_value_A))) / float(len(key_value_A))
        key_estimateB = float(len(set(key_value_B))) / float(len(key_value_B))
        if min(key_estimateA, key_estimateB) <= thres_prune:
            return 0

        mapped_keyA = list(SM.keys())

        if kyA not in self.query_fd:
            self.query_fd[kyA] = {}
            for idv, kv in enumerate(key_value_A):
                if kv not in self.query_fd[kyA]:
                    self.query_fd[kyA][kv] = []
                self.query_fd[kyA][kv].append(
                    ",".join(map(str, tableA[mapped_keyA].iloc[idv].tolist()))
                )
            fd = copy.deepcopy(self.query_fd[key])
        else:
            fd = copy.deepcopy(self.query_fd[key])

        mapped_keyB = list(SM.values())
        for idv, kv in enumerate(key_value_B):
            if kv not in fd:
                fd[kv] = []
            fd[kv].append(",".join(map(str, tableB[mapped_keyB].iloc[idv].tolist())))

        key_score = 0
        for fdk in fd.keys():
            key_score = key_score + float(len(set(fd[fdk]))) / float(
                tableA.shape[0] + tableB.shape[0]
            )

        return key_score

    def remove_dup(self, ranked_list, ks):
        """
        3要素リストの重複を取り除く．

        Args:
            ranked_list (list[any, any, any]): 重複を含んでいる可能性がある3要素リスト．
            ks (int): リストの最大サイズ.
        
        Returns:
            ranked_list (list[any, any, any]): 最大サイズksの重複を含まない3要素リスト．
        """
        res = []
        for i, j, l in ranked_list:
            flg = True
            for k, m in res: #i,l
                if self.real_tables[i].equals(self.real_tables[k]): #iとkのテーブルが同じとき
                    flg = False
                    break
            if flg:
                res.append((i, l))

            if len(res) == ks:
                break
        return res
    
    def remove_dup_with_scores(self, ranked_list, ks) -> list:# -> List[str, float, str]:
        """
        3要素リストの重複を取り除く．

        Args:
            ranked_list (list[any, any, any]): 重複を含んでいる可能性がある3要素リスト[table name, score, key_chosen].
            ks (int): リストの最大サイズ.
        
        Returns:
            ranked_list (list[any, any, any]): 最大サイズksの重複を含まない3要素リスト[table name, score, key_chosen]．
        """
        res = []
        for i, j, l in ranked_list:
            flg = True
            for k, m, n in res: #i,l
                if self.real_tables[i].equals(self.real_tables[k]): #iとkのテーブルが同じとき
                    flg = False
                    break
            if flg:
                res.append([i, j, l])

            if len(res) == ks:
                break
        return res


    def search_similar_tables_threshold2(
        self, query, beta, k, theta, thres_key_cache, thres_key_prune, tflag=False
    ):

        self.query = query
        self.query_fd = {}
        self.already_map = {}
        SM_test = SchemaMapping()

        start_time1 = timeit.default_timer()

        for i in self.schema_linking.keys():
            self.already_map[i] = {}

        query_col = self.sketch_query_cols(query)
        self.sketch_column_and_row_for_meta_mapping()#追加
        self.sample_rows_for_each_column()#追加

        
        time1 = 0
        start_time = timeit.default_timer()
        # Do mapping
        meta_mapping = SM_test.mapping_naive_tables(
            self.query,
            query_col,
            self.schema_element_sample_col, # self.schema_element_sample_col,
            self.schema_element_dtype, # self.schema_element_dtype,
        )
        end_time = timeit.default_timer()
        time1 += end_time - start_time
        
        
        # Compute unmatched pairs
        unmatched = {}
        for i in meta_mapping.keys():
            unmatched[i] = {}
            for j in query.columns.tolist(): # .tolist(): リスト型listに変換
                unmatched[i][j] = {}
                if (j in query_col) and (j not in meta_mapping[i]):
                    #for l in self.schema_element_sample_row[i].keys():
                    for l in self.schema_element_sample_row[i].keys():
                        unmatched[i][j][l] = ""


        top_tables = []
        Cache_MaxSim = {}

        rank2 = []
        rank_candidate = []


        #for i in self.real_tables.keys():
        for i in self.real_tables.keys():

            tname = i
            #gid = self.table_group[tname[6:]]
            gid = self.table_group[tname]
            if gid not in meta_mapping:
                continue

            #tableS = self.query
            tableS = query
            #tableR = self.real_tables[i]
            tableR = self.real_tables[i]

            start_time = timeit.default_timer()
            #SM, ms = self.schema_mapping(tableS, tableR, meta_mapping, gid)
            SM, ms = self.schema_mapping(tableS, tableR, meta_mapping, gid)
            end_time = timeit.default_timer()
            time1 = time1 + end_time - start_time
            Cache_MaxSim[tname] = ms

            if len(SM.items()) == 0:
                continue

            tableSnotintableR = []
            for sk in tableS.columns.tolist(): # .tolist(): リスト型listに変換
                if sk not in SM:
                    tableSnotintableR.append(sk)

            vname_score = float(1) / float(
                len(tableR.columns.values) + len(tableSnotintableR)
            )

            vname_score2 = float(
                min(len(tableS.columns.tolist()), len(tableR.columns.tolist())) - 1
            ) / float(len(tableR.columns.values) + len(tableSnotintableR) - 1)

            ubound = beta * vname_score2 + float(1 - beta) * Cache_MaxSim[tname]

            rank2.append(ubound)
            rank_candidate.append((tname, vname_score, SM))

        rank2 = sorted(rank2, reverse=True)
        rank_candidate = sorted(rank_candidate, key=lambda d: d[1], reverse=True)

        
        if len(rank_candidate) == 0:
            return []

        if len(rank_candidate) > k:
            ks = k
        else:
            ks = len(rank_candidate)
            
        for i in range(ks):
            #tableR = self.real_tables[rank_candidate[i][0]]
            tableR = self.real_tables[rank_candidate[i][0]]
            #gid = self.table_group[rank_candidate[i][0][6:]]
            gid = self.table_group[rank_candidate[i][0]]
            """
            SM_real = rank_candidate[i][2]
            (
                score,
                meta_mapping,
                unmatched,
                sm_time,
                key_chosen,
            ) = comp_table_similarity_key(
                SM_test,
                query, #self.query,
                tableR,#beta,
                SM_real,
                gid,
                meta_mapping,
                schema_linking, #self.schema_linking,
                thres_key_prune,
                thres_key_cache,
                already_map,
                query_fd,
            )
            """
            SM_real = rank_candidate[i][2]
            (
                col_sim,
                row_sim,
                meta_mapping,
                unmatched,
                sm_time,
                key_chosen,
            ) = self.comp_table_similarity_key(
                SM_test,
                query,
                tableR,
                SM_real,
                gid,
                meta_mapping,
                self.schema_linking, # self.schema_linking,
                thres_key_prune,
                thres_key_cache,
                unmatched,
            )
            score = beta * col_sim + float(1 - beta) * row_sim #追加
            #score = float(1 - beta) * col_sim + beta * row_sim #追加
            top_tables.append((rank_candidate[i][0], score, key_chosen))
            time1 += sm_time

        top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
        min_value = top_tables[-1][1]
        
        ks = ks - 1
        id = 0
        while True:
            if ks + id >= len(rank_candidate):
                break

            threshold = beta * rank_candidate[ks + id][1] + float(1 - beta) * rank2[0]

            if threshold <= min_value * theta:
                break
            else:
                id = id + 1
                if ks + id >= len(rank_candidate):
                    break

                #tableR = self.real_tables[rank_candidate[ks + id][0]]
                tableR = self.real_tables[rank_candidate[ks + id][0]]
                #gid = self.table_group[rank_candidate[ks + id][0][6:]]
                gid = self.table_group[rank_candidate[ks + id][0]]
                SM_real = rank_candidate[ks + id][2]
                (
                    col_sim,
                    row_sim,
                    meta_mapping,
                    unmatched,
                    sm_time,
                    key_chosen,
                    #rs,
                    #meta_mapping,
                    #unmatched,
                    #sm_time,
                    #key_chosen,
                ) = self.comp_table_similarity_key(
                    SM_test,
                    query,
                    tableR,
                    SM_real,
                    gid,
                    meta_mapping,
                    self.schema_linking, # self.schema_linking,
                    thres_key_prune,
                    thres_key_cache,
                    unmatched,
                    #SM_test,
                    #self.query,
                    #tableR,
                    #beta,
                    #SM_real,
                    #gid,
                    #meta_mapping,
                    #self.schema_linking,
                    #thres_key_prune,
                    #thres_key_cache,
                    #unmatched,
                )
                time1 += sm_time
                #new_score = rs
                new_score = beta * col_sim + float(1 - beta) * row_sim #追加

                if new_score <= min_value:
                    continue
                else:
                    top_tables.append(
                        (rank_candidate[ks + id][0], new_score, key_chosen)
                    )
                    top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
                    min_value = top_tables[ks][1]

        end_time1 = timeit.default_timer()
        time3 = end_time1 - start_time1

        logging.info("Schema Mapping Costs: %s Seconds" % time1)
        logging.info("Full Search Costs: %s Seconds" % time3)

        rtables_names = self.remove_dup(top_tables, ks)

        rtables = []
        for i, j in rtables_names:
            rtables.append((i, self.real_tables[i]))

        return rtables

    def search_similar_tables_threshold2_with_scores(
        self, query:pd.DataFrame, beta:float, k:int, theta:float, thres_key_cache:float, thres_key_prune:float, tflag:bool=False
    ) -> list:#-> List[str, pd.DataFrame, float]:

        self.query = query
        self.query_fd = {}
        self.already_map = {}
        SM_test = SchemaMapping()

        start_time1 = timeit.default_timer()

        for i in self.schema_linking.keys():
            self.already_map[i] = {}

        query_col = self.sketch_query_cols(query)
        self.sketch_column_and_row_for_meta_mapping()#追加
        self.sample_rows_for_each_column()#追加

        
        time1 = 0
        start_time = timeit.default_timer()
        # Do mapping
        meta_mapping = SM_test.mapping_naive_tables(
            self.query,
            query_col,
            self.schema_element_sample_col, # self.schema_element_sample_col,
            self.schema_element_dtype, # self.schema_element_dtype,
        )
        end_time = timeit.default_timer()
        time1 += end_time - start_time
        
        
        # Compute unmatched pairs
        unmatched = {}
        for i in meta_mapping.keys():
            unmatched[i] = {}
            for j in query.columns.tolist(): # .tolist(): リスト型listに変換
                unmatched[i][j] = {}
                if (j in query_col) and (j not in meta_mapping[i]):
                    #for l in self.schema_element_sample_row[i].keys():
                    for l in self.schema_element_sample_row[i].keys():
                        unmatched[i][j][l] = ""


        top_tables = []
        Cache_MaxSim = {}

        rank2 = []
        rank_candidate = []


        #for i in self.real_tables.keys():
        for i in self.real_tables.keys():

            tname = i
            #gid = self.table_group[tname[6:]]
            gid = self.table_group[tname]
            if gid not in meta_mapping:
                continue

            #tableS = self.query
            tableS = query
            #tableR = self.real_tables[i]
            tableR = self.real_tables[i]

            start_time = timeit.default_timer()
            #SM, ms = self.schema_mapping(tableS, tableR, meta_mapping, gid)
            SM, ms = self.schema_mapping(tableS, tableR, meta_mapping, gid)
            end_time = timeit.default_timer()
            time1 = time1 + end_time - start_time
            Cache_MaxSim[tname] = ms

            if len(SM.items()) == 0:
                continue

            tableSnotintableR = []
            for sk in tableS.columns.tolist(): # .tolist(): リスト型listに変換
                if sk not in SM:
                    tableSnotintableR.append(sk)

            vname_score = float(1) / float(
                len(tableR.columns.values) + len(tableSnotintableR)
            )

            vname_score2 = float(
                min(len(tableS.columns.tolist()), len(tableR.columns.tolist())) - 1
            ) / float(len(tableR.columns.values) + len(tableSnotintableR) - 1)

            ubound = beta * vname_score2 + float(1 - beta) * Cache_MaxSim[tname]

            rank2.append(ubound)
            rank_candidate.append((tname, vname_score, SM))

        rank2 = sorted(rank2, reverse=True)
        rank_candidate = sorted(rank_candidate, key=lambda d: d[1], reverse=True)

        
        if len(rank_candidate) == 0:
            return []

        if len(rank_candidate) > k:
            ks = k
        else:
            ks = len(rank_candidate)
            
        for i in range(ks):
            #tableR = self.real_tables[rank_candidate[i][0]]
            tableR = self.real_tables[rank_candidate[i][0]]
            #gid = self.table_group[rank_candidate[i][0][6:]]
            gid = self.table_group[rank_candidate[i][0]]
            SM_real = rank_candidate[i][2]
            (
                col_sim,
                row_sim,
                meta_mapping,
                unmatched,
                sm_time,
                key_chosen,
            ) = self.comp_table_similarity_key(
                SM_test,
                query,
                tableR,
                SM_real,
                gid,
                meta_mapping,
                self.schema_linking, # self.schema_linking,
                thres_key_prune,
                thres_key_cache,
                unmatched,
            )
            score = beta * col_sim + float(1 - beta) * row_sim #追加
            #score = float(1 - beta) * col_sim + beta * row_sim #追加
            top_tables.append((rank_candidate[i][0], score, key_chosen))
            time1 += sm_time

        top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
        min_value = top_tables[-1][1]
        
        ks = ks - 1
        id = 0
        while True:
            if ks + id >= len(rank_candidate):
                break

            threshold = beta * rank_candidate[ks + id][1] + float(1 - beta) * rank2[0]

            if threshold <= min_value * theta:
                break
            else:
                id = id + 1
                if ks + id >= len(rank_candidate):
                    break

                #tableR = self.real_tables[rank_candidate[ks + id][0]]
                tableR = self.real_tables[rank_candidate[ks + id][0]]
                #gid = self.table_group[rank_candidate[ks + id][0][6:]]
                gid = self.table_group[rank_candidate[ks + id][0]]
                SM_real = rank_candidate[ks + id][2]
                (
                    col_sim,
                    row_sim,
                    meta_mapping,
                    unmatched,
                    sm_time,
                    key_chosen,
                ) = self.comp_table_similarity_key(
                    SM_test,
                    query,
                    tableR,
                    SM_real,
                    gid,
                    meta_mapping,
                    self.schema_linking,
                    thres_key_prune,
                    thres_key_cache,
                    unmatched,
                )
                time1 += sm_time
                #new_score = rs
                new_score = beta * col_sim + float(1 - beta) * row_sim #追加

                if new_score <= min_value:
                    continue
                else:
                    top_tables.append(
                        (rank_candidate[ks + id][0], new_score, key_chosen)
                    )
                    top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
                    min_value = top_tables[ks][1]

        end_time1 = timeit.default_timer()
        time3 = end_time1 - start_time1

        logging.info("Schema Mapping Costs: %s Seconds" % time1)
        logging.info("Full Search Costs: %s Seconds" % time3)

        rtables_names = self.remove_dup_with_scores(top_tables, ks)

        rtables = []
        for i, j, l in rtables_names:
            rtables.append([i, self.real_tables[i], j])

        return rtables


    def calc_table_rel_using_juneau(self, tnameA:str, tableA:pd.DataFrame, k_juneau:int):
        """
        Juneauを使用してテーブルデータを<k_juneau>件検索し，その過程で得た類似度をself.calculated_simに格納する．
        """
        rtables=self.search_similar_tables_threshold2_with_scores(query=tableA, beta=0.1, k=k_juneau, theta=10, thres_key_cache=0.2, thres_key_prune=0.9, tflag=True)
        for tname, _, score in rtables:
            tgroup = self.table_group[tname]
            for tnameB, tgroupB in self.table_group.items():
                if tgroup == tgroupB:
                    self.cacheContentsSim(tnameA, tnameB, score)

    def calc_all_tables_rel_using_juneau(self, k_juneau:int=20):
        """
        クエリセット後かつ検索の実行前にこれを実行するとjuneauを適用できる．
        """
        self.flg_juneau=True
        for tnameA, tableA in self.query_table.items():
            #tableA = self.fetch_var_table(nname)
            self.calc_table_rel_using_juneau(tnameA,tableA, k_juneau)


    def search_using_only_juneau(self, k_juneau:int):
        """
        Juneauを使用してテーブルデータを上位<k_juneau>件検索する．テーブルデータの類似度は未ソートでself.nb_scoreに格納する．

        Args:
            k_juneau (int): Juneauのパラメータk．上位何件のテーブルデータを返すかの指定．
        """
        for tnameA, tableA in self.query_table.items():
            #tableA = self.fetch_var_table(nname)
            rtables=self.search_similar_tables_threshold2_with_scores(query=tableA, beta=0.1, k=k_juneau, theta=10, thres_key_cache=0.2, thres_key_prune=0.9, tflag=True)
            for tablename, _, score in rtables:
                nb_name = tablename[tablename.rfind("_")+1:]
                if nb_name not in self.nb_score:
                    self.nb_score[nb_name] = 0
                self.nb_score[nb_name] += score
           
    
    """人工データセット適用"""
    def load_artifical_dataset(self, dataset_size:int, change_id_list_path:str, change_libraries_list_path:str, retrieval_system_path:str) -> int:
        """
        人工データセット読み込み
        データセットのワークフローグラフself.Gに人工データセットを追加．
        人工データセットは，既存の111個のノートブックをコピーし，ノードを1個増減させたりライブラリを数個増減させることによって追加している．
        
        Args:
            dataset_size (int): 1以上100以下の整数パラメータ．データセットを何倍するかの指定．An int parameter to specify the number of times of dataset.
            change_id_list_path (str): 人工データセットについての情報が書かれたcsvのパス．
            change_libraries_list_path (str): ライブラリについての情報が書かれたchange_libraries_list.csvのパス．
            retrieval_system_path (str): dict_nb_name_and_cleaned_nb_nameなどを読むためのベースのパス．
        """
        if dataset_size==1:
            return 0
        self.flg_use_artifical_dataset=True
        self.load_libraries_of_artifical_dataset(dataset_size, change_libraries_list_path)
        self.load_dict_nb_name_and_cleaned_nb_name(retrieval_system_path)
        
        with open(change_id_list_path, mode="r") as f:
            read_lines=f.read()
        change_id_list=read_lines.split("\n")

        all_nodes_list = list(self.attr_of_db_nb_name.keys())
        all_edges_list = list(self.G.edges())
        valid_nb_name_append=self.valid_nb_name.append
        cell_num=0
        for n in all_nodes_list:
            if "cell" in n:
                cell_num = max(int(n[n.rfind("_")+1:]), cell_num)
        G_add_node=self.G.add_node
        G_add_edge=self.G.add_edge
        for copy_id in range(dataset_size-1):
            for cleaned_nb_name in self.valid_nb_name:
                node_correspondence_dict={}
                # cleaned_nb_nameの語尾がcp1などの場合，それは人工データなので除く
                if self.is_artifical_dataset(cleaned_nb_name):
                    continue                    

                new_cleaned_nb_name=f"{cleaned_nb_name}cp{copy_id}"
                v=self.dict_nb_name_and_cleaned_nb_name[cleaned_nb_name]
                new_nb_name = v[:v.rfind(".ipynb")] + f"_copy{copy_id}" + ".ipynb"
                valid_nb_name_append(new_cleaned_nb_name)
                if new_cleaned_nb_name not in self.dict_nb_name_and_cleaned_nb_name:
                    self.dict_nb_name_and_cleaned_nb_name[new_cleaned_nb_name] = new_nb_name
                if new_nb_name not in self.dict_nb_name_and_cleaned_nb_name2:
                    self.dict_nb_name_and_cleaned_nb_name2[new_nb_name] = new_cleaned_nb_name

                id_and_operation = []
                op=""
                for r in change_id_list:
                    if new_cleaned_nb_name in r:
                        id_and_operation = r.replace(" ","").split(",\t")
                        op = id_and_operation[2]
                        break

                if op == "delete":
                    delete_node = id_and_operation[3]
                    # 与えられたノートブック名の nodeを全て複製
                    for n in all_nodes_list:
                        if self.attr_of_db_nb_name[n] == cleaned_nb_name:
                            if n == delete_node:
                                continue
                            if self.attr_of_db_node_type[n] == "Cell":
                                cell_num+=1
                                node_name = f"cell_{cell_num}"
                                real_cell_id=self.attr_of_db_real_cell_id[n]
                                G_add_node(node_name, node_type="Cell", nb_name=new_cleaned_nb_name, real_cell_id=real_cell_id)
                                node_correspondence_dict[n]=node_name
                            elif self.attr_of_db_node_type[n] == "Var":
                                node_name=f"{n}cp{copy_id}"
                                G_add_node(node_name, node_type="Var", nb_name=new_cleaned_nb_name)
                                node_correspondence_dict[n]=node_name
                            elif self.attr_of_db_node_type[n] == "Display_data":
                                node_name=f"{n}cp{copy_id}"
                                display_type=self.attr_of_db_display_type[n]
                                G_add_node(node_name, node_type="Display_data", nb_name=new_cleaned_nb_name, display_type=display_type)
                                node_correspondence_dict[n]=node_name
                            else:
                                logging.error(f"err: ノードのタイプがいずれにも一致していない．node_name:{n}, nb_name:{cleaned_nb_name}") # -> 新しく追加したものがここに入っている?
                    for e in all_edges_list:
                        if delete_node not in e:
                            continue
                        if e[0] not in node_correspondence_dict or e[1] not in node_correspondence_dict:
                            continue
                        G_add_edge(node_correspondence_dict[e[0]], node_correspondence_dict[e[1]])
                elif op == "add":
                    selected_node = id_and_operation[3]
                    add_node = id_and_operation[4]
                    # 与えられたノートブック名の nodeを全て複製
                    for n in all_nodes_list:
                        if self.attr_of_db_nb_name[n] == cleaned_nb_name or n == add_node:
                            if self.attr_of_db_node_type[n] == "Cell":
                                cell_num+=1
                                node_name = f"cell_{cell_num}"
                                real_cell_id=self.attr_of_db_real_cell_id[n]
                                G_add_node(node_name, node_type="Cell", nb_name=new_cleaned_nb_name, real_cell_id=real_cell_id)
                                node_correspondence_dict[n]=node_name
                            elif self.attr_of_db_node_type[n] == "Var":
                                node_name=f"{n}cp{copy_id}"
                                G_add_node(node_name, node_type="Var", nb_name=new_cleaned_nb_name)
                                node_correspondence_dict[n]=node_name
                            elif self.attr_of_db_node_type[n] == "Display_data":
                                node_name=f"{n}cp{copy_id}"
                                display_type=self.attr_of_db_display_type[n]
                                G_add_node(node_name, node_type="Display_data", nb_name=new_cleaned_nb_name, display_type=display_type)
                                node_correspondence_dict[n]=node_name
                            else:
                                logging.error(f"err: ノードのタイプがいずれにも一致していない．node_name:{n}, nb_name:{cleaned_nb_name}") # -> 新しく追加したものがここに入っている?
                    for e in all_edges_list:
                        if e[0] not in node_correspondence_dict or e[1] not in node_correspondence_dict:
                            continue
                        G_add_edge(node_correspondence_dict[e[0]], node_correspondence_dict[e[1]])
                    G_add_edge(node_correspondence_dict[selected_node], node_correspondence_dict[add_node])

                else: #何も操作しない場合
                    # 与えられたノートブック名の nodeを全て複製
                    for n in all_nodes_list:
                        if self.attr_of_db_nb_name[n] == cleaned_nb_name:
                            if self.attr_of_db_node_type[n] == "Cell":
                                cell_num+=1
                                node_name = f"cell_{cell_num}"
                                real_cell_id=self.attr_of_db_real_cell_id[n]
                                G_add_node(node_name, node_type="Cell", nb_name=new_cleaned_nb_name, real_cell_id=real_cell_id)
                                node_correspondence_dict[n]=node_name
                            elif self.attr_of_db_node_type[n] == "Var":
                                node_name=f"{n}cp{copy_id}"
                                G_add_node(node_name, node_type="Var", nb_name=new_cleaned_nb_name)
                                node_correspondence_dict[n]=node_name
                            elif self.attr_of_db_node_type[n] == "Display_data":
                                node_name=f"{n}cp{copy_id}"
                                display_type=self.attr_of_db_display_type[n]
                                G_add_node(node_name, node_type="Display_data", nb_name=new_cleaned_nb_name, display_type=display_type)
                                node_correspondence_dict[n]=node_name
                            else:
                                logging.error(f"err: ノードのタイプがいずれにも一致していない．node_name:{n}, nb_name:{cleaned_nb_name}") # -> 新しく追加したものがここに入っている?
                    for e in all_edges_list:
                        if e[0] not in node_correspondence_dict or e[1] not in node_correspondence_dict:
                            continue
                        G_add_edge(node_correspondence_dict[e[0]], node_correspondence_dict[e[1]])

        self.attr_of_db_node_type=nx.get_node_attributes(self.G, "node_type")
        self.attr_of_db_nb_name=nx.get_node_attributes(self.G, "nb_name")
        self.attr_of_db_real_cell_id=nx.get_node_attributes(self.G, "real_cell_id")
        self.attr_of_db_display_type=nx.get_node_attributes(self.G, "display_type")
        self.set_all_label_node_list()
        self.set_nb_node_list()
        return 1


    def load_libraries_of_artifical_dataset(self, dataset_size:int, change_libraries_list_path:str):
        """
        人工データとして、change_libraries_list.csvをself.libraryに読み込む。
        dataset_sizeのみ取り出すのは時間がかかるので，現状は，dataset_size以上の数を読み込んでいる．

        Args:
            dataset_size (int): 1以上100以下の整数パラメータ．データセットを何倍するかの指定（現状未使用）．
            change_libraries_list_path (str): ライブラリについての情報が書かれたchange_libraries_list.csvのパス．
        """
        with open(change_libraries_list_path, mode="r") as f:
            load_contents=f.read()
        change_libraries_list=load_contents.split("\n")
        #chk_str = f"cp{dataset_size-1}"
        for r in change_libraries_list:
            r_list=r.split(",\t")
            if len(r_list) == 1:
                self.library[r_list[0]]=[]
            else:
                self.library[r_list[0]]=r_list[1:]

        """
        if dataset_size<=10:
            for r in change_libraries_list:
                r_list=r.split(",\t")
                name = r_list[0]
                if name[-3:] == chk_str:
                    break
                self.library[name]=r_list[1:]
        elif dataset_size<=100:
            for r in change_libraries_list:
                r_list=r.split(",\t")
                name = r_list[0]
                if name[-4:] == chk_str:
                    break
                self.library[name]=r_list[1:]
        elif dataset_size<=1000:
            for r in change_libraries_list:
                r_list=r.split(",\t")
                name = r_list[0]
                if name[-5:] == chk_str:
                    break
                self.library[name]=r_list[1:]
        """
        """
        for r in change_libraries_list:
            r_list=r.split(",\t")
            name = r_list[0]
            if dataset_size<10 and name[-3:] == chk_str:
                break
            elif dataset_size<100 and name[-4:] == chk_str:
                break
            elif dataset_size<1000 and name[-5:] == chk_str:
                break
            self.library[name]=r_list[1:]
        """


    def is_artifical_dataset(self, cleaned_nb_name:str) -> bool:
        """
        cleaned_nb_nameの語尾がcp1などの場合，それは人工データなので除く．

        Args:
            cleaned_nb_name (str): コピーを含む加工済みの計算ノートブック名．コピー元ならアンダーバーを取り除くなどしたもので，コピー後ならそれらにcp<ID>が末尾に付与されている．IDは整数．
        
        Returns:
            bool: Trueなら人工データセットでのみ含む計算ノートブック，Falseならオリジナルのデータセットに含まれる計算ノートブック．
        """
        if "cp" not in cleaned_nb_name:
            return False
        elif cleaned_nb_name[-1:] in NUM_STR_LIST:
            if cleaned_nb_name[-2:-1] in NUM_STR_LIST:
                if cleaned_nb_name[-3:-2] in NUM_STR_LIST and cleaned_nb_name[-5:-3] == "cp":
                    return True
                elif cleaned_nb_name[-4:-2] == "cp":
                    return True
            elif cleaned_nb_name[-3:-1] == "cp":
                return True
        return False


    def load_dict_nb_name_and_cleaned_nb_name(self, retrieval_system_path:str):
        """
        load_artifical_datasetで人工データセットを読み込む際に必要．
        """
        with open(f"{retrieval_system_path}/dict_nb_name_and_cleaned_nb_name.json", mode="r") as f:
            load_json=f.read()
            self.dict_nb_name_and_cleaned_nb_name=json.loads(load_json)
        with open(f"{retrieval_system_path}/dict_nb_name_and_cleaned_nb_name2.json", mode="r") as f:
            load_json=f.read()
            self.dict_nb_name_and_cleaned_nb_name2=json.loads(load_json)
        with open(f"{retrieval_system_path}/dict_nb_name_dir.json", mode="r") as f:
            load_json=f.read()
            self.dict_nb_name_dir=json.loads(load_json)
    


    # *** 以下，提案手法にjuneauを適用させるためのコード ***

    def init_query_col(self):
        """
        Juneauを使用したデータ類似度計算のための準備．
        新しいクエリになるたびに実行が必要．
        """
        self.query_col={}

    def set_query_col(self, query, query_table_name:str):
        self.query_col[query_table_name] = self.sketch_query_cols(query)

    def set_all_querytable_col(self):
        for tnameA, tableA in self.query_table.items():
            self.set_query_col(tableA, tnameA)

    def set_table_group2(self):
        """
        self.table_groupは{ノード名: グループID}なので，set_table_group2={グループID: list[ノード名]}に変換．
        """
        self.table_group2={}
        for tnameN, gid in self.table_group.items():
            if gid not in self.table_group2:
                self.table_group2[gid]=[]
            self.table_group2[gid].append(tnameN)

    def set_parameter(
        self,
        beta:float=0.1, 
        thres_key_cache:float=0.2, 
        thres_key_prune:float=0.9, 
    ):
        self.beta=beta
        self.thres_key_cache=thres_key_cache
        self.thres_key_prune=thres_key_prune

    def wrapper_calc_one_data_similarity_using_juneau(
        self, 
        tnameQ:str,
        tnameN:str, 
    ) -> float:
        """
        Juneauを使用し，指定した1つのテーブルに対してデータ類似度を計算し，その類似度を返す．

        事前にset_table_group2()の実行が必要．
        """
        score=self.calc_one_data_similarity_using_juneau(
            query=self.query_table[tnameQ],
            tnameQ=tnameQ,
            tnameN=tnameN, 
            beta=self.beta, 
            thres_key_cache=self.thres_key_cache, 
            thres_key_prune=self.thres_key_prune, 
        )
        # 同じスコアになるデータ類似度をキャッシュ
        if self.flg_running_faster["flg_caching"]:
            gid = self.table_group[tnameN]
            for tnameN in self.table_group2[gid]:
                self.cacheContentsSim(tnameQ, tnameN, score)
        return score


    def calc_one_data_similarity_using_juneau(
        self, 
        query,
        tnameQ:str,
        tnameN:str, 
        beta:float, 
        thres_key_cache:float, 
        thres_key_prune:float, 
    ) -> float:
        """
        Juneauを使用し，指定した1つのテーブルに対してデータ類似度を計算し，その類似度を返す．

        tableR: nb_nameのtable
        """
        # Original:search_similar_tables_threshold2_with_scores()

        gidN = self.table_group[tnameN]
        query_col=self.query_col[tnameQ]

        self.query = query
        self.query_fd = {}
        self.already_map = {}
        SM_test = SchemaMapping()

        start_time1 = timeit.default_timer()

        for i in self.schema_linking.keys():
            self.already_map[i] = {}

        #self.query_col = self.sketch_query_cols(query) # -> 重複を防ぐため先にset_query_col()であらかじめ取得．
        #self.sketch_column_and_row_for_meta_mapping()#追加
        self.sketch_column_and_row_for_meta_mapping_for_one_notebook(gidN)
        #self.sample_rows_for_each_column()#追加
        self.sample_rows_for_each_column_for_one_notebook(gidN)

        
        time1 = 0
        start_time = timeit.default_timer()
        # Do mapping
        meta_mapping = SM_test.mapping_naive_tables(
            self.query,
            query_col,
            self.schema_element_sample_col, # データベースのテーブルに関する情報なので，クエリに依らない
            self.schema_element_dtype, # データベースのテーブルに関する情報なので，クエリに依らない
        )
        end_time = timeit.default_timer()
        time1 += end_time - start_time
        
        
        # Compute unmatched pairs
        unmatched = {}
        for i in meta_mapping.keys():
            unmatched[i] = {}
            for j in self.query.columns.tolist(): # .tolist(): リスト型listに変換
                unmatched[i][j] = {}
                if (j in query_col) and (j not in meta_mapping[i]):
                    for l in self.schema_element_sample_row[i].keys():
                        unmatched[i][j][l] = ""


        top_tables = []
        Cache_MaxSim = {}

        rank2 = []
        rank_candidate = []


        #for gid in self.real_tables.keys():

        tableS = self.query
        tableR = self.real_tables[tnameN]

        start_time = timeit.default_timer()
        SM, ms = self.schema_mapping(tableS, tableR, meta_mapping, gidN)
        end_time = timeit.default_timer()
        time1 = time1 + end_time - start_time
        Cache_MaxSim[tnameN] = ms

        if len(SM.items()) == 0:
            return 0

        """
        tableSnotintableR = []
        for sk in tableS.columns.tolist(): # .tolist(): リスト型listに変換
            if sk not in SM:
                tableSnotintableR.append(sk)

        vname_score = float(1) / float(
            len(tableR.columns.values) + len(tableSnotintableR)
        )

        vname_score2 = float(
            min(len(tableS.columns.tolist()), len(tableR.columns.tolist())) - 1
        ) / float(len(tableR.columns.values) + len(tableSnotintableR) - 1)

        ubound = beta * vname_score2 + float(1 - beta) * Cache_MaxSim[tnameN]

        rank2.append(ubound)
        rank_candidate.append((tnameN, vname_score, SM))

        rank2 = sorted(rank2, reverse=True)
        rank_candidate = sorted(rank_candidate, key=lambda d: d[1], reverse=True)

        
        if len(rank_candidate) == 0:
            return []

        if len(rank_candidate) > k:
            ks = k
        else:
            ks = len(rank_candidate)
        """
            
        #for i in range(ks):
        #tableR = self.real_tables[rank_candidate[gidN][0]]
        #gid = self.table_group[rank_candidate[i][0][6:]]
        #gid = self.table_group[rank_candidate[i][0]]
        #SM_real = rank_candidate[gidN][2]
        SM_real = SM

        (
            col_sim,
            row_sim,
            meta_mapping,
            unmatched,
            sm_time,
            key_chosen,
        ) = self.comp_table_similarity_key(
            SM_test,
            self.query,
            tableR,
            SM_real,
            gidN,
            meta_mapping,
            self.schema_linking, # self.schema_linking,
            thres_key_prune,
            thres_key_cache,
            unmatched,
        )

        time1 += sm_time
        end_time1 = timeit.default_timer()
        time3 = end_time1 - start_time1

        logging.info("Juneau --- Schema Mapping Costs: %s Seconds" % time1)
        logging.info("Juneau --- Full Search Costs: %s Seconds" % time3)
        return beta * col_sim + float(1 - beta) * row_sim 

        """
        score = beta * col_sim + float(1 - beta) * row_sim #追加
        #score = float(1 - beta) * col_sim + beta * row_sim #追加
        top_tables.append((rank_candidate[i][0], score, key_chosen))
        time1 += sm_time

        top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
        min_value = top_tables[-1][1]
        
        ks = ks - 1
        id = 0
        while True:
        
            if ks + id >= len(rank_candidate):
                return 0

            threshold = beta * rank_candidate[ks + id][1] + float(1 - beta) * rank2[0]

            if threshold <= min_value * theta:
                return 0
            else:
                id = id + 1
                if ks + id >= len(rank_candidate):
                    return 0

                #tableR = self.real_tables[rank_candidate[ks + id][0]]
                tableR = self.real_tables[rank_candidate[ks + id][0]]
                #gid = self.table_group[rank_candidate[ks + id][0][6:]]
                gid = self.table_group[rank_candidate[ks + id][0]]
                SM_real = rank_candidate[ks + id][2]
                (
                    col_sim,
                    row_sim,
                    meta_mapping,
                    unmatched,
                    sm_time,
                    key_chosen,
                ) = self.comp_table_similarity_key(
                    SM_test,
                    self.query,
                    tableR,
                    SM_real,
                    gid,
                    meta_mapping,
                    self.schema_linking,
                    thres_key_prune,
                    thres_key_cache,
                    unmatched,
                )
                time1 += sm_time
                #new_score = rs
                new_score = beta * col_sim + float(1 - beta) * row_sim #追加

                if new_score <= min_value:
                    continue
                else:
                    top_tables.append(
                        (rank_candidate[ks + id][0], new_score, key_chosen)
                    )
                    top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
                    min_value = top_tables[ks][1]

        end_time1 = timeit.default_timer()
        time3 = end_time1 - start_time1

        logging.info("Schema Mapping Costs: %s Seconds" % time1)
        logging.info("Full Search Costs: %s Seconds" % time3)

        #rtables_names = self.remove_dup_with_scores(top_tables, ks)

        #rtables = []
        #for i, j, l in rtables_names:
        #    rtables.append([i, self.real_tables[i], j])

        #return rtables
        return new_score
        """

    # データベースのテーブルに対して，列をサンプリング
    # sketch_column_and_row_for_meta_mappingを1つのテーブルグループに適用する．
    def sketch_column_and_row_for_meta_mapping_for_one_notebook(self, gid:int, sz:int=5, row_size:int=1000):
        """
        データベースのテーブルに対して，列をサンプリングする．

        具体的には，self.schema_elementについて，各テーブルグループに対して列数および行数が指定の最大数以下となるように，
        テーブルグループごとにデータをサンプリングし，結果をself.schema_element_sample_colに格納．

        self.schema_element_sample_col={}の定義が事前に必要．
        """
        if gid not in self.schema_element_sample_col: # 同じ操作を重複して実行しないため（備考：self.schema_element_sample_colはクエリに依らない）
            #self.schema_element_sample_col = {}
            #for i in self.schema_element.keys(): # 各テーブルグループに対し以下の処理
            i = gid
            self.schema_element_sample_col[i] = {}
            if len(self.schema_element[i].keys()) <= sz: # テーブルグループの列数がsz個以下の時
                # 表データの行のサンプリング
                # scはschemaの略か.
                for sc in self.schema_element[i].keys(): # 指定テーブルグループの各列に対して
                    if len(self.schema_element[i][sc]) < row_size: # 最大行数よりもサンプリング対象の行数が少ない場合は, すべての行を得る.
                        self.schema_element_sample_col[i][sc] = self.schema_element[i][sc]
                    else: # 最大行数よりもサンプリング対象の行数が多い場合は, 最大行数分だけをランダムに選んで抽出する.
                        self.schema_element_sample_col[i][sc] = random.sample(
                            self.schema_element[i][sc], row_size
                        )
            else: # サンプリング対象の列数がsz列より多いとき
                ##
                # sc_choice ([str, float]): strは列名, floatはその列のデータ(値)の集合に対して, 
                # |セット集合|/|多重集合|を計算している.(セット集合 = 重複無し集合). 
                # scはschemaの略か.
                ##
                sc_choice = []
                for sc in self.schema_element[i].keys():
                    if sc == "Unnamed: 0" or "index" in sc:
                        continue
                    if self.schema_element_dtype[i][sc] is np.dtype(float):
                        continue
                    sc_value = list(self.schema_element[i][sc]) # テーブルグループIDがi, 列名がscの実データ値をsc_valueに格納
                    # sc列目のデータ集合に対し, |セット集合|/|多重集合|を計算
                    sc_choice.append(
                        (sc, float(len(set(sc_value))) / float(len(sc_value)))
                    )
                # その列の値ができるだけバラバラな値をとる順にソートする．
                sc_choice = sorted(sc_choice, key=lambda d: d[1], reverse=True) #降順

                count = 0
                for sc, v in sc_choice:
                    if count == sz:
                        break
                    # 表データの行のサンプリング
                    if len(self.schema_element[i][sc]) < row_size: # 最大行数よりもサンプリング対象の表データの行数が少ない場合は, すべての行を得る.
                        self.schema_element_sample_col[i][sc] = self.schema_element[i][
                            sc
                        ]
                    else: # 最大行数よりもサンプリング対象の表データの行数が多い場合は, 最大行数分だけをランダムに選んで抽出する.
                        self.schema_element_sample_col[i][sc] = random.sample(
                            self.schema_element[i][sc], row_size
                        )

                    count += 1


    # データベースのテーブルに対して，行をサンプリング
    def sample_rows_for_each_column_for_one_notebook(self, gid:int, row_size:int=1000):
        """
        データベースのテーブルに対して，行をサンプリングする．

        具体的には，関数sample_rows_for_each_column()を1つのテーブルグループに（i = gidとして）適用し，
        self.schema_element_sample_row[gid]に格納する．
        """
        if gid not in self.schema_element_sample_row: # 同じ操作を重複して実行しないため（備考：self.schema_element_sample_rowはクエリに依らない）
            self.schema_element_sample_row[gid] = {}
            for sc in self.schema_element[gid].keys(): #各列に対して(sc:列名)
                if len(self.schema_element[gid][sc]) < row_size: # サンプリング前のデータ数（行数）がrow_sizeより小さければそのままを保持
                    self.schema_element_sample_row[gid][sc] = self.schema_element[gid][sc]
                else: #サンプリング前のデータ数（行数）がrow_sizeより大きれば, 行をサンプリングして行数をrow_sizeにする．
                    self.schema_element_sample_row[gid][sc] = random.sample(
                        self.schema_element[gid][sc], row_size
                    )


    def levenshtein_dist(self, w_c) -> dict:
        """
        セルの塊ごとにコードを比較．最も高い類似度のセルのコードとの類似度をクエリの各セルに対し計算し，その合計値をそのノートブックのスコアとする．
        最後に0以上1以下の値に正規化される．
        """
        sim_list={}
        n_q_num=len(self.query_table)
        #node_db_list=list(self.G.nodes)
        node_db_list=self.all_node_list["Cell"]
        for n_q in self.query_table:
            sim_list[n_q]={}
            for n_db in node_db_list:
                if self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    continue
                nb_name=self.attr_of_db_nb_name[n_db]
                if nb_name not in self.valid_nb_name:
                    continue
                calc_code_start_time = timeit.default_timer()
                sim=CodeSimilarity.code_sim_based_on_levenshtein_dist(n_q, n_db)
                calc_code_end_time = timeit.default_timer()
                self.each_calc_time_sum["Cell"]+=calc_code_end_time-calc_code_start_time
                if nb_name not in sim_list:
                    sim_list[n_q][nb_name]=0
                sim_list[n_q][nb_name]=max(sim_list[n_q][nb_name], sim)

        n_q_num=len(self.query_table)
        notebooks_code_sim={}
        for nb_name in self.valid_nb_name:
            notebooks_code_sim[nb_name]=0
            for n_q in self.query_table:
                notebooks_code_sim[nb_name]+=sim_list[n_q][nb_name]
            notebooks_code_sim[nb_name]*=w_c
            notebooks_code_sim[nb_name]/=n_q_num
        return notebooks_code_sim


    def existing_method_sum_using_juneau(self, if_codeSimPerCell=False):
        """
        Set-based search method w/o optimization.
        Save computational notebook similarity in self.nb_score.

        ノード数による調整あり．
        if_codeSimPerCell=False: コードは全体的な類似性を見る．
        if_codeSimPerCell=True: コードはセル単位で局所的な類似性を見る．
        """
        self.init_each_calc_time_sum()
        self.init_arranged_all_weights()
        self.nb_score={}
        table_q_num=len(self.query_table)
        code_q_num=len(self.query_cell_code)
        nb_score_according_to_table_sim={}
        nb_score_according_to_code_sim={}

        if self.w_v!=0:
            nb_score_according_to_table_sim=self.existing_method_calc_table_sim_using_juneau()
        if self.w_c!=0:
            if if_codeSimPerCell:
                nb_score_according_to_code_sim=self.set_based_method_calc_code_sim()
            else:
                nb_score_according_to_code_sim=self.existing_method_calc_code_sim2()

        for nb_name in self.valid_nb_name:
            if nb_name not in nb_score_according_to_table_sim:
                nb_score_according_to_table_sim[nb_name]=0
            if nb_name not in nb_score_according_to_code_sim:
                nb_score_according_to_code_sim[nb_name]=0
            self.nb_score[nb_name]=0
            if table_q_num!=0:
                self.nb_score[nb_name] += nb_score_according_to_table_sim[nb_name] * self.w_v / table_q_num
            if code_q_num!=0:
                self.nb_score[nb_name] += nb_score_according_to_code_sim[nb_name] * self.w_c / code_q_num


    
    def data_profiling(self, datanode_cur):
        flg_profiled = False
        data_cur=self.fetch_var_table(datanode_cur)
        max_dataid = 0
        for dataid, datanodelist in self.data_profile.items():
            comparenode = datanodelist[0]
            data_compare=self.fetch_var_table(comparenode)
            if data_compare.equals(data_cur):
                self.data_profile[dataid].append(datanode_cur)
                flg_profiled = True
            if flg_profiled:
                break
            max_dataid = max(max_dataid, dataid)
        if not flg_profiled:
            self.data_profile[max_dataid+1]=[datanode_cur]
            
    def profile_all_data(self, cachemode="small"):
        """
        データにIDを付与して，IDごとにそのデータを使用している計算ノートブック名をストアする（キャッシュ）
        """
        def getOriginalDataList():
            originalDataList = []
            checked_nb = set()
            cellnode_and_id = sorted(self.attr_of_db_node_type, key=lambda d: d[1], reverse=True) #降順
            for cellnode in cellnode_and_id:
            #for cellnode, cellID in self.attr_of_db_node_type.items():
                if self.attr_of_db_nb_name[cellnode] in checked_nb:
                    continue
                for childnode in self.G.successors(cellnode):
                    if self.attr_of_db_node_type[childnode] == "Var":
                        originalDataList.append(childnode)
                        checked_nb.add(self.attr_of_db_nb_name[cellnode])
            return originalDataList

        if cachemode == "small":
            originalDataList = getOriginalDataList()
            for node in originalDataList:
                self.data_profiling(datanode_cur=node)
        elif cachemode == "all":
            for node, nodetype in self.attr_of_db_node_type.items():
                if nodetype == "Var":
                    self.data_profiling(datanode_cur=node)

    def WrapperOfDataProfile(self, cachemode="small"):
        self.data_profile={}
        starttime = timeit.default_timer()
        self.profile_all_data(cachemode)
        endtime = timeit.default_timer()
        logging.info(endtime - starttime)
        logging.info("Seconds")
        logging.info((endtime - starttime)/60)
        logging.info("Minutes")
        if cachemode == "small":
            small_data_profile = self.data_profile
        elif cachemode == "all":
            all_data_profile = self.data_profile

    def transformdict2df(self, data_profile):
        # 全データの方でも2秒ぐらいで終わる
        colname = ["DataID", "DataName"]
        storedataframe = pd.DataFrame(index=[], columns=colname)

        for k, datalist in data_profile.items():
            for dataname in datalist:
                appenddf = pd.DataFrame([[k, dataname]], columns=colname)
                storedataframe = storedataframe.append(appenddf, ignore_index=True)
        return storedataframe

    def savedataID(self, data_profile, cachemode="small"):
        with self.postgres_eng.connect() as conn:
            conn.execute(
                f"CREATE SCHEMA IF NOT EXISTS datacache;" 
        )
        storedataframe = self.transformdict2df(data_profile)
        
        if cachemode == "small":
            storename = "idtodatanamesmall"
        elif cachemode == "all":
            storename = "idtodataname"
        
        try:
            storedataframe.to_sql(
                name=storename,
                con=self.postgres_eng.connect(),
                schema=config.sql.datacache, 
                if_exists="replace", #既にあった場合は置き換え
                index=False,
            )
        except Exception as e:
            logging.error(f"Unable to store 'datacache' due to error {e}")

    def readDataCache(self, cachemode="small"):
        """
        cachemode: 'small' or 'all'
        return dict{data name: data ID}
        """
        if cachemode == "small":
            with self.postgres_eng.connect() as conn:
                readdataIDlist = pd.read_sql_table(f"idtodatanamesmall", conn, schema=f"datacache")
        elif cachemode == "all":
            with self.postgres_eng.connect() as conn:
                readdataIDlist = pd.read_sql_table(f"idtodataname", conn, schema=f"datacache")
        else:
            logging.info("error: failed to get cache for dataID")
            
        datanametodataID = {}
        for row in readdataIDlist.itertuples():
            datanametodataID[row.DataName] = row.DataID

        return datanametodataID

    def cacheContentsSim(self, querynode, targetnode, similarityscore):
        """
        if self.flg_running_faster["flg_caching"]:
            self.calculated_sim[(nodenameQ, nodenameN)]=sim
        を以下に変更すると同一．
        self.cacheContentsSim(nodenameQ, nodenameN, sim)
        """
        if self.flg_running_faster["flg_caching"]:
            if self.flg_useDataCache and targetnode in self.datanametodataID and not self.flg_juneau:
                    self.calculated_sim[(querynode, self.datanametodataID[targetnode])]=similarityscore
            else:
                self.calculated_sim[(querynode, targetnode)]=similarityscore

    def loadContentsSim(self, querynode, targetnode):
        """
        if (nodenameQ, nodenameN) in self.calculated_sim:
            return self.calculated_sim[(nodenameQ, nodenameN)]
        を以下に変更すると同一．
        sim = self.loadContentsSim(nodenameQ, nodenameN)
        if sim is not None:
            return sim
        """
        if self.flg_running_faster["flg_caching"]:
            if (querynode, targetnode) in self.calculated_sim:
                return self.calculated_sim[(querynode, targetnode)]
            elif self.flg_useDataCache and targetnode in self.datanametodataID and (querynode, self.datanametodataID[targetnode]) in self.calculated_sim:
                return self.calculated_sim[(querynode, self.datanametodataID[targetnode])]
        return None

