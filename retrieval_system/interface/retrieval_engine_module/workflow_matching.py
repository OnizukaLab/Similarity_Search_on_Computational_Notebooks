# Copyright 2020 Juneau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module to store a table's provenance.
"""

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
import timeout_decorator
import matplotlib.pyplot as plt

sys.path.append('/Users/runa/Desktop/大学/4年/実装/my_code/juneau_copy')
sys.path.append('/Users/runa/Desktop/大学/4年/実装/my_code/core-detection_copy/lib')
from py2neo import Node, Relationship, NodeMatcher, RelationshipMatcher
from juneau.config import config
from juneau.utils.funclister import FuncLister
from juneau.db.table_db import generate_graph, pre_vars
from juneau.search.search_prov_code import ProvenanceSearch
from mymodule.code_relatedness import CodeRelatedness
#from lib import CodeComparer


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
    def __init__(self, postgres_eng, graph_eng, sim_col_thres=0.5, w_c=1, w_v=1, w_l=1, w_d=1, k=6, change_next_node_thres=0.8, valid_nb_name_file_path="../データセット/valid_nb_name.txt"):#, thres_data_profile=0.9):
        """
        PostgreSQLのインスタンスとpy2neoのインスタンスをインスタンスにセットする．

        Args:
            postgres_eng: PostgreSQLのインスタンス
            graph_eng (Graph): py2neoのクラス`Graph`のインスタンス．インスタンス変数graph_dbに格納．
            query (list[Node]): クエリのワークフローグラフ．ラベルは"Cell"がセルノード，"Var"が表形式データのノード，"OneWildcard"は1つのいずれかのノード，"AnyWildcard"は0個以上のいずれかのノード
        """
        self.graph_db = graph_eng
        self.postgres_eng = postgres_eng
        self.query_workflow = [] #query_nodeの名前の方がいいかも
        self.query_relationship={}
        self.query_data = []
        self.query_display_type=[]
        self.nb_node_dict={} # dict{str: list[Node]}: {NB name: [Nodes in the NB]}, initialized by "fetch_db_node"
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
        self.graphs_dependencies={}
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
        self.top_k_score=1001001
        self.flg_running_faster={}
        self.class_code_relatedness=CodeRelatedness()
        self.invalid_by_workflow_structure={"invalid":set(), "valid":set()}
        self.calc_v_count=0
        #self.data_profile={}
        #self.thres_data_profile=thres_data_profile
        self.element_profile={}
        self.nb_node_list={}
        self.init_each_calc_time_sum()

        self.query_acol_set={}
        self.query_scma_dict={}
        self.calc_v_microbenchmark={"load":0, "set":0, "column_rel":0, "sort":0, "table_rel":0}



    def init_each_calc_time_sum(self):
        self.each_calc_time_sum={"Cell":0.0, "Var":0.0, "Display_data":0.0, "Library":0.0}

    def set_each_w(self, w_c=None, w_v=None, w_l=None, w_d=None):
        if not w_c is None:
            logging.info(f"reset code sim weight {self.w_c} --> {w_c}")
            self.w_c=w_c
        if not w_v is None:
            logging.info(f"reset table data sim weight {self.w_v} --> {w_v}")
            self.w_v=w_v
        if not w_l is None:
            logging.info(f"reset library sim weight {self.w_l} --> {w_l}")
            self.w_l=w_l
        if not w_d is None:
            logging.info(f"reset display output sim weight {self.w_d} --> {w_d}")
            self.w_d=w_d

    def set_k(self,k):
        self.k=k


    def set_query_attr(self):
        self.attr_of_q_node_type = nx.get_node_attributes(self.QueryGraph, "node_type")
        self.attr_of_q_real_cell_id=nx.get_node_attributes(self.QueryGraph, "real_cell_id")
        self.attr_of_q_display_type=nx.get_node_attributes(self.QueryGraph, "display_type")
        self.set_query_workflow_info_Display_data()


    def set_sim_col_thres(self, sim_col_thres):
        self.sim_col_thres=sim_col_thres
    
    # 読み込み
    # クエリでは不使用　実装の際の便宜用
    def load_calculated_sim(self, calculated_sim_path):
        with open(calculated_sim_path, mode="r") as f:
            load_json=f.read()

        load_json=json.loads(load_json)
        calculated_sim={}
        for list_row in load_json:
            calculated_sim[(list_row[0], list_row[1])]=list_row[2]
        self.calculated_sim=calculated_sim

    # 書き込み
    # クエリでは不使用　実装の際の便宜用
    def store_calculated_sim(self, calculated_sim_path):
        store_json=[]
        for key, val in self.calculated_sim.items():
            store_json.append([key[0], key[1], val])
        store_json=json.dumps(store_json)
        with open(calculated_sim_path, mode="w") as f:
            f.write(store_json)

    
    def fetch_db_node(self):
        """
        Neo4Jに格納されている全てのノードを取り出す．
        """
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
        cell_node_list = matcher.match("Cell").all()
        var_node_list = matcher.match("Var").all()
        root_count=0

        def append_node2nb_node_dict(nb_name, node):
            if not nb_name in self.nb_node_dict:
                self.nb_node_dict[nb_name]=[]
            self.nb_node_dict[nb_name].append(node)

        #セルノードのnb名
        for node in cell_node_list:
            nb_name=node["nb_name"]
            append_node2nb_node_dict(nb_name, node)
            node_name=node["name"]
            try:
                if (r_matcher.match((node,), r_type="Parent").first()) is None:
                    if node not in self.root_list:
                        if node["real_cell_id"] != 1:
                            self.root_list2.append([node["name"], node["nb_name"], node["real_cell_id"]])
                        self.root_list.append(node)
                    #logging.info(f"node {node_name} in nb {nb_name} is a root node.")
                    root_count+=1
            except:
                pass

        #変数ノードのnb名
        for node in var_node_list:
            node_name=node["name"]
            nb_name=node_name[node_name.rindex("_")+1:]
            #print(nb_name)
            append_node2nb_node_dict(nb_name, node)

        exist_nb_name=[]
        for i in self.nb_node_dict:
            for n in self.nb_node_dict[i]:
                if n in self.root_list:
                    exist_nb_name.append(i)
        for i in self.nb_node_dict:
            if i not in exist_nb_name:
                logging.info(i)
        logging.info(f"root_count: {root_count}, nb_name count: {len(self.nb_node_dict)}")
        print(self.root_list2)

    def add_code_node(self, source_code):
        cell_id=len(self.query_cell_code)+1
        #ノード追加
        self.QueryGraph.add_node(f"cell_query_{cell_id}", node_type="Cell", real_cell_id=f"{cell_id}")
        #エッジ追加
        if cell_id > 1:
            self.QueryGraph.add_edge(f"cell_query_{cell_id-1}", f"cell_query_{cell_id}")
        #実際のソースコードを保持
        self.query_cell_code[f"cell_query_{cell_id}"]=source_code




    def make_sample_query(self, query_nb_name):
        cid=1
        data_id=1
        filepath="../../データセット/query_sample/"+query_nb_name
        if not os.path.exists(filepath):
            raise Exception("File does not exist: "+filepath)
        prev_node=None
        
        # inform notebook file
        with open(filepath,"r") as r:
            nb = nbf.read(r,as_version=3) # nbconvert format version = 3
            for x in nb.worksheets:
                for cell in x.cells:
                    if cell.cell_type =="code":
                        if not cell.language == 'python':
                            raise ValueError('Code must be in python!')
                        if cell.input.strip() != "":
                            cell_code=self.cleaning_one_cell_code(cell.input.split("\n"))
                            self.query_cell_code[cid]=cell_code
                            new_cell_node=Node("Cell", real_cell_id=cid+1)
                            if len(self.query_workflow)==0:
                                self.query_root=new_cell_node
                            if prev_node != None:
                                if prev_node not in self.query_relationship:
                                    self.query_relationship[prev_node]=[]
                                self.query_relationship[prev_node].append(new_cell_node)
                            self.query_workflow.append(new_cell_node)
                            #print(self.query_cell_code[cid])
                            dep, _1, _2 =self.__parse_code(cell_code)
                            for j in dep:
                                if type(dep[j][0][0]) is tuple:
                                    print(dep[j][0][0][0])
                                    new_var_node=Node("Var", var_name=dep[j][0][0][0], data_id=data_id)
                                    data_id+=1
                                    self.query_relationship[new_cell_node].append(new_var_node)
                                else:
                                    print(dep[j][0][0])
                                    new_var_node=Node("Var", var_name=dep[j][0][0], data_id=data_id)
                                    data_id+=1
                                    self.query_relationship[new_cell_node].append(new_var_node)
                                if len(dep[j][1])==0:
                                    continue
                                if type(dep[j][1][0]) is tuple:
                                    pass
                                    #print(dep[j][1][0][0])
                                else:
                                    pass
                                    #print(dep[j][1][0])
                                #new_var_node = Node("Var", var_name)
                                #self.query_relationship[new_cell_node] = new_var_node
                            prev_node=new_cell_node
                    cid+=1
    
    def make_sample_query_nx1_2_old(self):
        self.QueryGraph=nx.DiGraph()

        # 頂点のセット
        for i in range(1,4):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")

        # 辺のセット
        self.QueryGraph.add_edge("cell_query_1", "cell_query_2", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "query_var1", edge_type="Contains")

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
        self.query_workflow_info={"Cell": 3, "Var": 1, "Display_data": {}, "max_indegree": 1, "max_outdegree": 2}

    def make_sample_query_nx1_2(self):
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
    def make_sample_query_nx2_3_old(self):
        self.QueryGraph=nx.DiGraph()

        for i in range(1,5):
            self.QueryGraph.add_node(f"cell_query_{i}", node_type="Cell", real_cell_id=f"{i}")
        self.QueryGraph.add_node("query_var1", node_type="Var", data_type="pandas.core.frame.DataFrame")
        #参考: self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        self.QueryGraph.add_node("query_display2", node_type="Display_data", display_type="DataFrame", cell_id="2")
        self.QueryGraph.add_node("query_display1", node_type="Display_data", display_type="DataFrame", cell_id="4")

        self.QueryGraph.add_edge("cell_query_1", "cell_query_2", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "cell_query_3", edge_type="Successor")
        self.QueryGraph.add_edge("cell_query_2", "query_display2", edge_type="Display")
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
        self.query_workflow_info={"Cell": 4, "Var": 1, "Display_data": {"DataFrame": 2}, "max_indegree": 1, "max_outdegree": 3}

    #実際のNBの一部
    # ユーザ実験に利用
    def make_sample_query_nx2_3(self):
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
    def __parse_code(self, code_list):
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
    def add_node_to_graph(self, node):
        """
        add node to DiGraph 'self.G'

        Args:
            node (Node): py2neo instance.
        """
        err=False
        if node.has_label("Var"):
            nb_name=node["name"]
            nb_name=nb_name[nb_name.rfind("_")+1:]
            if nb_name not in self.valid_nb_name:
                return True
            self.G.add_node(node["name"], node_type="Var", nb_name=nb_name, data_type=node["data_type"])
        elif node.has_label("Cell"):
            if node["nb_name"] not in self.valid_nb_name:
                return True
            self.G.add_node(node["name"], node_type="Cell", nb_name=node["nb_name"], real_cell_id=node["real_cell_id"])
        elif node.has_label("Display_data"):
            if node["nb_name"] not in self.valid_nb_name:
                return True
            self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        return err
    
    # 使用
    def add_node_to_graph_to_knn_graph(self, node):
        """
        add node to DiGraph 'self.KnnGraph'

        Args:
            node (Node): py2neo instance.
        """
        err=False
        if node.has_label("Var"):
            nb_name=node["name"]
            nb_name=nb_name[nb_name.rfind("_")+1:]
            if nb_name not in self.valid_nb_name:
                return True
            self.KnnGraph.add_node(node["name"], node_type="Var", nb_name=nb_name, data_type=node["data_type"])
        elif node.has_label("Cell"):
            if node["nb_name"] not in self.valid_nb_name:
                return True
            self.KnnGraph.add_node(node["name"], node_type="Cell", nb_name=node["nb_name"], real_cell_id=node["real_cell_id"])
        elif node.has_label("Display_data"):
            if node["nb_name"] not in self.valid_nb_name:
                return True
            self.KnnGraph.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["real_cell_id"])
        return err

    def init_db_workflow_info(self, nb_name):
        if nb_name not in self.db_workflow_info:
            self.db_workflow_info[nb_name]={"Cell": 0, "Var": 0, "Display_data": {}, "max_indegree": 0, "max_outdegree": 0}

    # set_db_graphで使用
    def add_node_to_graph_and_set_workflow_info(self, node):
        """
        add node to DiGraph 'self.G'

        Args:
            node (Node): py2neo instance.
        """
        err=False
        if node.has_label("Var"):
            nb_name=node["name"]
            nb_name=nb_name[nb_name.rfind("_")+1:]
            if nb_name not in self.valid_nb_name:
                return True
            self.G.add_node(node["name"], node_type="Var", nb_name=nb_name, data_type=node["data_type"])
            self.init_db_workflow_info(nb_name)
            self.db_workflow_info[nb_name]["Var"]+=1
        elif node.has_label("Cell"):
            if node["nb_name"] not in self.valid_nb_name:
                return True
            self.G.add_node(node["name"], node_type="Cell", nb_name=node["nb_name"], real_cell_id=node["real_cell_id"])
            self.init_db_workflow_info(nb_name)
            self.db_workflow_info[nb_name]["Cell"]+=1
        elif node.has_label("Display_data"):
            if node["nb_name"] not in self.valid_nb_name:
                return True
            self.G.add_node(node["name"], node_type="Display_data", nb_name=node["nb_name"], display_type=node["data_type"], cell_id=node["cell_id"])
            self.init_db_workflow_info(nb_name)
            if node in self.attr_of_q_display_type:
                display_type = self.attr_of_q_display_type[node]
                self.countup_display_data(nb_name, display_type)
        return err

    # グラフをデータベースから読み込むために使用
    def set_db_graph(self):
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

        for start_node in node_list:
            if not self.G.has_node(start_node["name"]):
                continue
            rel_list = r_matcher.match((start_node, ), "Successor").all() + r_matcher.match((start_node, ), "Contains").all() + r_matcher.match((start_node, ), "Usedby").all() + r_matcher.match((start_node, ), "Display").all()
            for rel in rel_list:
                end_node=rel.end_node
                if not self.G.has_node(end_node["name"]):
                    continue
                #    err=self.add_node_to_graph(end_node)
                #    if err:
                #        continue
                self.G.add_edge(start_node["name"], end_node["name"])

        self.attr_of_db_node_type=nx.get_node_attributes(self.G, "node_type")
        self.attr_of_db_nb_name=nx.get_node_attributes(self.G, "nb_name")
        self.attr_of_db_real_cell_id=nx.get_node_attributes(self.G, "real_cell_id")
        self.attr_of_db_display_type=nx.get_node_attributes(self.G, "display_type")
        self.set_all_label_node_list()
        self.set_nb_node_list()

    # ベンチマーク用: すでに読み込んだグラフから読み込むために使用
    def set_db_graph2(self, graph):
        self.set_valid_nb_name()

        self.G = graph
        self.attr_of_db_node_type=nx.get_node_attributes(self.G, "node_type")
        self.attr_of_db_nb_name=nx.get_node_attributes(self.G, "nb_name")
        self.attr_of_db_real_cell_id=nx.get_node_attributes(self.G, "real_cell_id")
        self.attr_of_db_display_type=nx.get_node_attributes(self.G, "display_type")
        self.set_all_label_node_list()
        self.set_nb_node_list()
    
    def set_nb_node_list(self):
        for node in self.attr_of_db_nb_name:
            nb_name = self.attr_of_db_nb_name[node]
            node_type=self.attr_of_db_node_type[node]
            if nb_name not in self.nb_node_list:
                self.nb_node_list[nb_name]={"Cell":[], "Var":[], "Display_data":[]}
            self.nb_node_list[nb_name][node_type].append(node)

    # とりあえず不使用
    #def store_knn_graph_via_neo4j(self, knn_k=5):
    def append_edge_to_neo4j_as_knn_graph_via_neo4j(self, knn_k=5):
        """
        neo4jにストアする場合
        """
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
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
                    ### knnグラフとなるように辺を追加 ###
                    #self.G.add_edge(nodeA, nodeB, edge_type="Sim")
                    nodeA = matcher.match(name=node_nameA).first()
                    nodeB = matcher.match(name=sorted_rel_list[i][0]).first()
                    sim_edge=r_matcher.match((nodeA, nodeB), "Sim").first()
                    if sim_edge is None:
                        sim_edge = Relationship(nodeA, "Sim", nodeB, sim_val=sorted_rel_list[i][1])
                        self.graph_db.create(sim_edge)
                        self.graph_db.push(sim_edge)
        
    # とりあえず不使用
    def fetch_knn_graph_via_neo4j(self):
        """
        neo4jにストアする場合
        """
        self.KnnGraph = nx.DiGraph()
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
        node_list = matcher.match().all()

        for start_node in node_list: #全てのノードをnetworkxのDiGraphに追加
            err=self.add_node_to_graph_to_knn_graph(start_node)
            #err=self.add_node_to_graph_and_set_workflow_info(start_node)
            if err:
                continue
        for start_node in node_list: #全てのノードをnetworkxのDiGraphに追加
            if not self.KnnGraph.has_node(start_node["name"]):
                continue
            rel_list = r_matcher.match((start_node, ), "Sim").all()
            for rel in rel_list:
                end_node=rel.end_node
                if not self.KnnGraph.has_node(end_node["name"]):
                    continue
                self.KnnGraph.add_edge(start_node["name"], end_node["name"])
                if "sim_val" not in rel:
                    continue
                self.calculated_sim[(start_node["name"], end_node["name"])]=rel["sim_val"]

    
    def init_table_knn_sim_list(self, table_name):
        with self.psql_engine.connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {config.sql.knn_sim_list}.knn_sim_list " 
                f"(start_node TEXT, end_node TEXT, sim );" # VARCHAR(M)は, 最大M文字数の可変長文字列の型.
            )

    def store_knn_graph_via_sql(self, knn_k=5):
        """
        sqlでストアする場合
        """
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
                    
        
    # とりあえず不使用
    def fetch_knn_graph(self):
        self.KnnGraph = nx.DiGraph()
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
        node_list = matcher.match().all()

        for start_node in node_list: #全てのノードをnetworkxのDiGraphに追加
            err=self.add_node_to_graph_to_knn_graph(start_node)
            #err=self.add_node_to_graph_and_set_workflow_info(start_node)
            if err:
                continue
        for start_node in node_list: #全てのノードをnetworkxのDiGraphに追加
            if not self.KnnGraph.has_node(start_node["name"]):
                continue
            rel_list = r_matcher.match((start_node, ), "Sim").all()
            for rel in rel_list:
                end_node=rel.end_node
                if not self.KnnGraph.has_node(end_node["name"]):
                    continue
                self.KnnGraph.add_edge(start_node["name"], end_node["name"])
                if "sim_val" not in rel:
                    continue
                self.calculated_sim[(start_node["name"], end_node["name"])]=rel["sim_val"]

    # 使用することになりそう
    def set_element_profile(self, thres_ele_profile=0.7):
        self.thres_ele_profile=thres_ele_profile
        self.element_profile={}
        if not self.all_node_list:
            self.set_all_label_node_list()
        for node_type in self.all_node_list:
            first_flg=True
            #print(node_type)
            node_list=self.all_node_list[node_type]
            rel_list=[]
            for nodeA in node_list:
                if first_flg:
                    self.element_profile[nodeA]={"node_type": node_type, "contents": None, "sim_node": []}
                    first_flg=False
                    continue
                if nodeA in self.element_profile:
                    continue
                if len(self.element_profile) == 0:
                    self.element_profile[nodeA]={"node_type": node_type, "contents": None, "sim_node": []}
                    #self.fetch_node_contents()
                #nodeB_list=self.element_profile[self.element_profile["node_type"]==node_type].values
                #for nodeB in nodeB_list:
                for nodeB in self.element_profile:
                    if self.element_profile[nodeB]["node_type"]!=node_type:
                        continue
                    rel_list.append((nodeB, self.calc_rel_between_db_node_with_timecount(nodeA,  nodeB)))
                    sorted_rel_list=sorted(rel_list, key=lambda x:x[1], reverse=True)
                    if sorted_rel_list[0][1] < self.thres_ele_profile:
                        self.element_profile[nodeA]={"node_type": node_type, "contents": None, "sim_node": []}
                    else:
                        self.element_profile[sorted_rel_list[0][0]]["sim_node"].append(nodeA)

    # 使用することになりそう
    def set_element_profile_with_node_type_old(self, node_type, thres_ele_profile=0.7):
        self.thres_ele_profile=thres_ele_profile
        self.element_profile={}
        if not self.all_node_list:
            self.set_all_label_node_list()
        if node_type in self.all_node_list:
            node_list=self.all_node_list[node_type]
            for nodeA in node_list:
                rel_list=[]
                if nodeA in self.element_profile:
                    continue
                if len(self.element_profile) == 0:
                    self.element_profile[nodeA]={"node_type": node_type, "contents": None, "sim_node": []}
                    continue
                    #self.fetch_node_contents()
                #nodeB_list=self.element_profile[self.element_profile["node_type"]==node_type].values
                #for nodeB in nodeB_list:
                for nodeB in self.element_profile:
                    if self.element_profile[nodeB]["node_type"]!=node_type:
                        continue
                    rel_list.append((nodeB, self.calc_rel_between_db_node_with_timecount(nodeA,  nodeB)))
                if len(rel_list)==0:
                    continue
                sorted_rel_list=sorted(rel_list, key=lambda x:x[1], reverse=True)
                if sorted_rel_list[0][1] < self.thres_ele_profile:
                    self.element_profile[nodeA]={"node_type": node_type, "contents": None, "sim_node": []}
                else:
                    self.element_profile[sorted_rel_list[0][0]]["sim_node"].append(nodeA)

    # 使用することになりそう
    def set_element_profile_with_node_type(self, node_type, thres_ele_profile=0.7):
        self.thres_ele_profile=thres_ele_profile
        self.element_profile[node_type]={}
        if not self.all_node_list:
            self.set_all_label_node_list()
        if node_type in self.all_node_list:
            node_list=self.all_node_list[node_type]
            for nodeA in node_list:
                rel_list=[]
                if nodeA in self.element_profile[node_type]:
                    continue
                if len(self.element_profile[node_type]) == 0:
                    self.element_profile[node_type][nodeA]={"contents": None, "sim_node": []}
                    continue
                    #self.fetch_node_contents()
                #nodeB_list=self.element_profile[node_type][self.element_profile[node_type]["node_type"]==node_type].values
                #for nodeB in nodeB_list:
                for nodeB in self.element_profile[node_type]:
                    insert_tuple=(nodeB, self.calc_rel_between_db_node_with_timecount(nodeA, nodeB))
                    rel_list.append(insert_tuple)
                    logging.info(f"({nodeA},{insert_tuple[0]}) --- {insert_tuple[1]}")
                if len(rel_list)==0:
                    continue
                sorted_rel_list=sorted(rel_list, key=lambda x:x[1], reverse=True)
                if sorted_rel_list[0][1] < self.thres_ele_profile:
                    self.element_profile[node_type][nodeA]={"contents": None, "sim_node": []}
                else:
                    self.element_profile[node_type][sorted_rel_list[0][0]]["sim_node"].append(nodeA)
        else:
            logging.info(f"invalid node type: {node_type}")


    def set_all_label_node_list(self):
        self.all_node_list={"Cell": [], "Var": [], "Display_data": []}
        node_list = self.attr_of_db_node_type
        ret_node_list=[]
        for node in node_list:
            node_type=self.attr_of_db_node_type[node]
            self.all_node_list[node_type].append(node)

    def countup_display_data(self, nb_name, display_type):
        if display_type not in self.db_workflow_info[nb_name]["Display_data"]:
            self.db_workflow_info[nb_name]["Display_data"][display_type]=0
        self.db_workflow_info[nb_name]["Display_data"][display_type]+=1
        
    def get_db_workflow_info(self):
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
            

    def get_sta_of_dataset_old(self):
        self.get_db_workflow_info()
        self.set_all_label_node_list()
        num_of_nb=len(self.db_workflow_info)
        num_of_node={"max":{}, "min":{}, "avg":{}, "sum":0}
        sta_of_table_data={"max":0, "min":0, "avg":0, "sum":0}
        num_of_node["max"]={"All":0, "Cell":0, "Var":0, "Display_data":0}
        num_of_node["sum"]={"All":0, "Cell":0, "Var":0, "Display_data":0}
        num_of_node["min"]={"All":1001001, "Cell":1001001, "Var":1001001, "Display_data":1001001}
        num_of_node["avg"]={}
        num_of_node["avg"]["All"]=len(self.G.nodes)/num_of_nb
        for node_type in self.all_node_list:
            num_of_node["avg"][node_type]=len(self.all_node_list[node_type])/num_of_nb

        for nb_name in self.db_workflow_info:
            num_of_any_type_node=0
            for node_type in self.all_node_list:
                num_of_node["min"][node_type]=min(self.db_workflow_info[nb_name][node_type], num_of_node["min"][node_type])
                num_of_node["max"][node_type]=max(self.db_workflow_info[nb_name][node_type], num_of_node["max"][node_type])
                num_of_any_type_node+=self.db_workflow_info[nb_name][node_type]
                num_of_node["sum"][node_type]
            #num_of_node["min"][node_type]=min(num_of_any_type_node, num_of_node["min"][node_type])
            #num_of_node["max"][node_type]=max(num_of_any_type_node, num_of_node["max"][node_type])
        for var_name in self.all_node_list["Var"]:
            table_size = self.get_table_size3(self.fetch_var_table(var_name))
            sta_of_table_data["min"] = min(table_size, sta_of_table_data["min"])
            sta_of_table_data["max"] = max(table_size, sta_of_table_data["max"])
            sta_of_table_data["sum"] += table_size
        sta_of_table_data["avg"]=sta_of_table_data["sum"]/num_of_nb

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

    def get_sta_of_query_source_code(self):# 書き途中
        stack_n=[self.query_root]
        query_code_list=[]
        while stack_n != []:
            node=stack_n.pop()
            if self.attr_of_q_node_type[node] == "Cell":
                query_code_list.append(self.query_cell_code[node])
            stack_n.extend(list(self.QueryGraph.successors(node)))

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
    def subgraph_matching_with_wildcard_in_query(self, limit=None, nb_limit=None, tflg=False):
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


    def chk_invalid_by_workflow_structure_old(self, nb_name):
        """
        return True if num of any workflow components is less than query
        """
        if nb_name in list(self.invalid_by_workflow_structure["invalid"]):
            return True
        for ele in list(self.query_workflow_info):
            if ele not in self.db_workflow_info[nb_name]:
                #logging.info(f"'{ele}' not in self.db_workflow_info['{nb_name}']")
                #return True
                continue
            if ele == "Display_data":
                for display_type in self.query_workflow_info[ele]:
                    if display_type not in self.db_workflow_info[nb_name][ele]:
                        #logging.info(f"'{display_type}' not in self.db_workflow_info['{nb_name}']['{ele}']")
                        self.invalid_by_workflow_structure["invalid"].add(nb_name)
                        return True
                    elif self.query_workflow_info[ele][display_type] > self.db_workflow_info[nb_name][ele][display_type]:
                        #logging.info(f"self.query_workflow_info['{ele}']['{display_type}'] > self.db_workflow_info['{nb_name}']['{ele}']['{display_type}']")
                        self.invalid_by_workflow_structure["invalid"].add(nb_name)
                        return True
            elif self.query_workflow_info[ele] > self.db_workflow_info[nb_name][ele]:
                #logging.info(f"self.query_workflow_info['{ele}'] > self.db_workflow_info['{nb_name}']['{ele}']")
                self.invalid_by_workflow_structure["invalid"].add(nb_name)
                return True
        return False
        
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
    def real_subgraph_matching_part_with_timecount_old(self, limit=None, nb_limit=None):
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
        start_time1 = timeit.default_timer()
        time2=0
        n_count2=0
        time3=0
        n_count3=0
        self.detected_count=0
        
        visited=[]
        matched=[]
        count=0 #あとで消す
        nb_name_list=[] #あとで消す

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
            # *********以下 あとで消す***************************************************
            count+=1 #あとで消す
            if (not limit is None) and count>=limit: #あとで消す
                break #あとで消す
            if  (not nb_limit is None) and len(self.ans_list) >= nb_limit: #あとで消す
                break #あとで消す
            if nb_name not in nb_name_list: #あとで消す
                nb_name_list.append(nb_name) #あとで消す
                #logging.info(f"nb name: {nb_name}") #あとで消す
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

            self.flg_detection=False
            
            start_time2 = timeit.default_timer()
            detected_subgraph=self.rec_new4(ret_list, self.query_root, node_db, visited, matched, nb_name)
            end_time2 = timeit.default_timer()
                
            #print("detected subgraph:", detected_subgraph)
            if self.flg_detection:
                n_count2+=1
                time2+=(end_time2-start_time2)
            else:
                n_count3+=1
                time3+=(end_time2-start_time2)
        end_time1 = timeit.default_timer()
        time1 = end_time1 - start_time1

        #print("")
        #print("-------- ans_list --------")
        #print(self.ans_list)

        logging.info("subgraph matching completed!")
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
        logging.info(f"calculated num of nb: {len(nb_name_list)}")
        if len(nb_name_list) != 0:
            logging.info(f"time par nb: {time1/len(nb_name_list)}")
            logging.info(f"num of detected subgraph par nb: {self.detected_count/len(nb_name_list)}")
        return time1,time2,n_count2,time3,n_count3,self.detected_count, len(nb_name_list)
 
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
    def real_subgraph_matching_part_weaving_with_timecount(self, limit=None, nb_limit=None, flg_knn=False):
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
        start_time1 = timeit.default_timer()
        time2=0
        n_count2=0
        time3=0
        n_count3=0
        self.detected_count=0
        
        visited=[]
        matched=[]
        count=0 #あとで消す
        #nb_name_list=[] #あとで消す

        #有効なnb名をセット
        self.set_valid_nb_name()

        query_label=self.attr_of_q_node_type[self.query_root]
        node_db_list=[]
        if query_label in ["Cell", "Var", "Display_data"]:
            node_db_list=self.all_node_list[query_label]
            #for node_name, node_label in self.attr_of_db_node_type.items():
            #    if node_label == query_label:
            #        node_db_list.append(node_name)
        else: # クエリのrootがOneWildcardまたはAnyWildcardのとき
            node_db_list=self.G.nodes() # list[str]

        visited_db=[]


        for node_db in node_db_list:
            if node_db in visited_db:
                continue
            # node_db (str): データベースのノートブックをワークフロー化したグラフのノード
            nb_name=self.attr_of_db_nb_name[node_db]
            if nb_name not in self.valid_nb_name:
                continue
            if self.flg_running_faster["chk_invalid_by_workflow_structure"] and self.chk_invalid_by_workflow_structure(nb_name):
                continue
            # *********以下 あとで消す***************************************************
            #if (not limit is None) and count>=limit: #あとで消す
            #    break #あとで消す
            #if  (not nb_limit is None) and len(self.ans_list) > nb_limit: #あとで消す
            #    break #あとで消す
            #if nb_name not in nb_name_list: #あとで消す
            #    nb_name_list.append(nb_name) #あとで消す
                #logging.info(f"nb name: {nb_name}") #あとで消す
            # *********以上 あとで消す***************************************************

            
            ret_list=[]
            #logging.info("detecting subgraph...")
            roop_count=0
            matched=[]
            visited=[]
            if nb_name not in self.ans_list:
                self.ans_list[nb_name]=[]
            if nb_name not in self.nb_score:
                self.nb_score[nb_name]=0.0

            lib_rel=self.calc_lib_rel_with_timecount(nb_name)
            
            current_score=0
            # self.calculated_sim に計算結果を保存してあとから参照できるので，ここでの計算はあとで活用できるためそこまで時間のロスにはならないはず
            if self.flg_prune_under_sim([self.query_root], lib_rel+self.calc_rel_with_timecount(self.query_root, node_db)):
                continue


            # knnグラフ構成後
            next_node_list=[node_db]

            for node_db2 in next_node_list:
                if node_db2 in visited_db:
                    continue
                self.flg_detection=False

                
                #_ =self.rec_new4_weaving(ret_list, self.query_root, node_db2, visited, matched, nb_name, current_score=0)
                #if self.subgraph_matching_mode==2:
                #    start_time2 = timeit.default_timer()
                #    _ =self.rec_new4_weaving2(ret_list, self.query_root, node_db2, visited, matched, nb_name, current_score=0)
                #    end_time2 = timeit.default_timer()
                #if self.subgraph_matching_mode==3:
                #    start_time2 = timeit.default_timer()
                #    _ =self.rec_new4_weaving3(ret_list, self.query_root, node_db2, visited, matched, nb_name, current_score=self.calc_rel_with_timecount(self.query_root, node_db))
                #    end_time2 = timeit.default_timer()
                #if self.flg_rec_chk:
                #    start_time2 = timeit.default_timer()
                #    _ =self.rec_new4_weaving3_chk(ret_list, self.query_root, node_db2, visited, matched, nb_name, current_score=lib_rel+self.calc_rel_with_timecount(self.query_root, node_db))
                #    end_time2 = timeit.default_timer()
                #else:
                if self.subgraph_matching_mode==3:
                    start_time2 = timeit.default_timer()
                    _ =self.rec_new4_weaving3(ret_list, self.query_root, node_db2, visited, matched, nb_name, current_score=lib_rel+self.calc_rel_with_timecount(self.query_root, node_db))
                    end_time2 = timeit.default_timer()
                elif self.subgraph_matching_mode==4:
                    start_time2 = timeit.default_timer()
                    _ =self.rec_new4_weaving4(ret_list, self.query_root, node_db2, visited, matched, nb_name)
                    end_time2 = timeit.default_timer()
                visited_db.append(node_db2)

                if self.flg_detection:
                    n_count2+=1
                    time2+=(end_time2-start_time2)
                    count+=1 #あとで消す
                else:
                    n_count3+=1
                    time3+=(end_time2-start_time2)

                if flg_knn and self.calc_rel_with_timecount(self.query_root, node_db2) > self.change_next_node_thres:
                    next_node_list.extend(self.KnnGraph.successors(node_db2))

        end_time1 = timeit.default_timer()
        time1 = end_time1 - start_time1

        #print("")
        #print("-------- ans_list --------")
        #print(self.ans_list)
        nb_count=len(self.ans_list)

        logging.info("subgraph matching completed!")
        logging.info(f"time of getting node list and subgraph matching: {time1}")
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
    
    # 交互実行のアルゴリズムでのみ使用．他はcalc_lib_scoreを利用している．
    def calc_lib_rel_with_timecount(self, nb_name):
        calc_start_time = timeit.default_timer()
        ret=self.calc_lib_rel(nb_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        return ret

    # 交互実行のアルゴリズムでのみ使用．他はcalc_lib_scoreを利用している．
    def calc_lib_rel(self, nb_name):
        if self.w_l==0:
            return 0
        if nb_name in self.calculated_lib_sim:
            return self.calculated_lib_sim[nb_name] * self.w_l

        lib_list=self.fetch_all_library_from_db(nb_name)
        lib_rel=self.calc_lib_score(self.query_lib, lib_list)
        self.calculated_lib_sim[nb_name]=lib_rel
        return lib_rel * self.w_l

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


    def set_flg_running_faster(self, flg_chk_invalid_by_workflow_structure=True, flg_flg_prune_under_sim=True, flg_optimize_calc_order=True, flg_caching=True, flg_calc_data_sim_approximately=False, flg_cache_query_table=False):
        self.flg_running_faster["chk_invalid_by_workflow_structure"]=flg_chk_invalid_by_workflow_structure
        self.flg_running_faster["flg_prune_under_sim"]=flg_flg_prune_under_sim
        self.flg_running_faster["flg_optimize_calc_order"]=flg_optimize_calc_order
        self.flg_running_faster["flg_caching"]=flg_caching
        self.flg_running_faster["flg_calc_data_sim_approximately"]=flg_calc_data_sim_approximately
        self.flg_running_faster["flg_cache_query_table"]=flg_cache_query_table
        

    # 以前のバージョン(クエリによってはエラー)
    def rec_new4(self, ret_list, snode_q, snode_db, visited, matched, nb_name):
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

        
        if self.attr_of_db_node_type[snode_db] != self.attr_of_q_node_type[snode_q]:
            logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[snode_q]}, {self.attr_of_db_node_type[snode_db]}")
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
                # ここの条件になる場合は存在しないはず
                logging.info("*************detected duplicate.*****************")
                logging.info(new_matched)
            return ret_list, new_matched


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
                return ret_list, matched
            for n in list(self.QueryGraph.successors(snode_q)):
                if n in new_visited:
                    continue
                n_q=n
                break
        if n_q is None:
            return ret_list, new_matched


        flg_not_found=True
        for n_db in next_db_list:
            if (n_q, n_db) in new_matched:
                continue

            if self.attr_of_q_node_type[snode_q]=="AnyWildcard":
                pass
            elif self.attr_of_q_node_type[n_q]==self.attr_of_db_node_type[n_db]:
                pass
            else:
                continue

            flg_not_found=False

            flg_exist_any_n_q=False
            remaining_node_list=[]
            for n in next_q_list:
                if n is n_q:
                    continue
                if self.attr_of_q_node_type[n]=="AnyWildcard":
                    continue
                if n not in new_visited:
                    remaining_node_list.append(n)
                    flg_exist_any_n_q=True

            if flg_exist_any_n_q:
                new_new_visited=list(set(new_visited+[n_q]))
                if self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[n_q]}, {self.attr_of_db_node_type[n_db]}")
                #new_new_matched=list(set(new_matched+[(n_q, n_db)]))
                new_new_matched=new_matched+[(n_q, n_db)]
                ret_list, matched2=self.rec_new4(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name) 
                ret_list, _ =self.rec_new4(ret_list, n_q, n_db, new_new_visited, matched2, nb_name) 
            else:
                ret_list, _ =self.rec_new4(ret_list, n_q, n_db, new_visited, new_matched, nb_name)
        if flg_not_found:
            return ret_list, matched
        return ret_list, new_matched

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
    
    # *** 作成中 ***
    def rec_with_wildcard_in_query_old(self, ret_list, snode_q, snode_db, visited, matched, nb_name):
        """
        wildcardを含むクエリに対応．
        閉路を持つクエリでは誤ったサブグラフが検出される可能性あり．

        Args:
            parent_node_is_wildcard (bool): True if parent node in query graph is wildcard.

        current_node_is_wildcard (bool): True if current node in query graph is wildcard.
        """
        #ret_list: 現在探しているマッピングに対して，現在の深さのノード以外もマッピングしているノード間の関係を全て入れたもの
        print(snode_q,snode_db)

        # 初期化
        next_q_list=[]
        next_db_list=[]
        #current_node_is_wildcard=False

        
        if (self.attr_of_q_node_type[snode_q] not in ["OneWildcard", "AnyWildcard"]) and self.attr_of_db_node_type[snode_db] != self.attr_of_q_node_type[snode_q]:
            logging.info(f"error1: not match node type of workflow. {self.attr_of_q_node_type[snode_q]}, {self.attr_of_db_node_type[snode_db]}")
        #new_matched=list(set(matched+[(snode_q,snode_db)]))
        if self.attr_of_q_node_type[snode_q]!="AnyWildcard":
            new_matched=matched+[(snode_q,snode_db)]
            new_visited=list(set(visited+[(snode_q)]))
        else: #よくなさそう
            new_matched=matched
            new_visited=visited

        print("new_visited", new_visited)
        # クエリの全てのノードに訪れた場合，終了
        remaining_list=[]
        flg_exist_any_n_q=False
        for n in self.QueryGraph.nodes():
            if self.attr_of_q_node_type[n]=="AnyWildcard":
                continue
            if n not in new_visited:
                remaining_list.append(n)
                #print(f"remaining: {n}")
                flg_exist_any_n_q=True
                #break
        print(f"remaining list: {remaining_list}")
        #if len(new_visited) == len(self.QueryGraph.nodes): # クエリの全てのノードに訪れたかどうかの判定
        if not flg_exist_any_n_q:
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
            next_q_list.extend(list(self.QueryGraph.successors(snode_q)))
            for n in next_q_list:
                if n in new_visited:
                    continue
                n_q=n
                break
            next_q_list.append(snode_q)
            #next_db_list.append(snode_db)
            #n_q=snode_q
            #current_node_is_wildcard=True
        else:
            # クエリNBのノードsnode_qの子ノードのリストを得る
            next_q_list.extend(list(self.QueryGraph.successors(snode_q)))
            #if len(next_db_list) < len(next_q_list):
            #    return ret_list, matched, visited
            for n in next_q_list:
                if self.attr_of_q_node_type[n]!="AnyWildcard":
                    continue
                n_q=n
                break
            for n in next_q_list:
                if self.attr_of_q_node_type[n]=="AnyWildcard":
                    continue
                if n in new_visited:
                    continue
                n_q=n
                break
        if n_q is None:
            return ret_list, new_matched, new_visited

        if self.attr_of_q_node_type[snode_q]=="AnyWildcard" and self.attr_of_q_node_type[n_q]!="AnyWildcard":
            next_db_list.append(snode_db)

        if self.attr_of_q_node_type[n_q]=="AnyWildcard":
            new_visited_tmp=list(set(new_visited))
            new_matched_tmp=list(set(new_matched))

        flg_not_found=True
        for n_db in next_db_list:
            if (n_q, n_db) in new_matched:
                continue

            if self.attr_of_q_node_type[n_q]=="AnyWildcard":
                new_visited=list(set(new_visited_tmp))
                new_matched=list(set(new_matched_tmp))
                #new_visited=new_visited_tmp
                #new_matched=new_matched_tmp
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
                    ret_list, matched2, visited2=self.rec_with_wildcard_in_query(ret_list, n_q, n_db, new_visited, new_matched, nb_name) 
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


    # *** 完成版　メインで使用　ワイルドカード不問 ***
    def rec_with_wildcard_in_query_old2(self, ret_list, snode_q, snode_db, visited, matched, nb_name):
        """
        wildcardを含むクエリに対応．
        閉路を持つクエリでは誤ったサブグラフが検出される可能性あり．

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
                if self.attr_of_db_node_type[n]=="Cell":
                    next_db_list.append(n)
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
    
    # 作り途中
    def chk_exist_any_n_q(self, next_q_list, current_n_q, visited):
        flg_exist_any_n_q=False
        w_count=0
        for n in next_q_list:
            if n is current_n_q:
                continue
            if self.attr_of_q_node_type[n]=="AnyWildcard":
                w_count+=1
                continue
            if n not in visited:
                #remaining_node_list.append(n)
                flg_exist_any_n_q=True
        return flg_exist_any_n_q

    # *** 旧バージョン 提案手法  メインで使用(一括でサブグラフマッチング) エラーあり ***
    def rec_new4_weaving_old(self, ret_list, snode_q, snode_db, visited, matched, nb_name, current_score):
        """
        提案手法
        self.Gを利用して一括でサブグラフマッチング．(先にデータベースから一括でロード)
        クエリグラフはself.QueryGraph
        閉路を持つクエリでは誤ったサブグラフが検出される可能性あり．

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

        # クエリのノードsnode_qの子ノードのリストを得る
        
        if self.attr_of_db_node_type[snode_db] != self.attr_of_q_node_type[snode_q]:
            logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[snode_q]}, {self.attr_of_db_node_type[snode_db]}")
        
        # 類似度下限にかかっているかをチェック(下限より低ければPruning)
        current_score+=self.calc_rel_with_timecount(snode_q, snode_db)
        new_visited=list(set(visited+[(snode_q)]))
        if self.flg_prune_under_sim(new_visited, current_score):
            return ret_list, matched
        new_matched=matched+[(snode_q,snode_db)]


        # クエリの全てのノードに訪れた場合，終了
        if len(new_visited) == len(self.QueryGraph.nodes): # クエリの全てのノードに訪れたかどうかの判定
            new_matched=list(set(new_matched))
            new_matched.sort()
            if new_matched not in ret_list:
                ret_list.append(new_matched)
                if nb_name not in self.nb_score or self.nb_score[nb_name] < current_score:
                    self.nb_score[nb_name]=current_score
                #print(new_matched)
                self.flg_detection=True
                self.ans_list[nb_name].append(new_matched)
                self.detected_count+=1
            else:
                # ここの条件になる場合は存在しないはず
                logging.info("*************detected duplicate.*****************")
                logging.info(new_matched)
            return ret_list, new_matched


        # データNBのノードsnode_dbの子ノードのリストを得る
        next_db_list=list(self.G.successors(snode_db))


        #if snode_q.has_label("AnyWildcard"):
        n_q=None
        if self.attr_of_q_node_type[snode_q]=="AnyWildcard":
            next_q_list.append(snode_q)
            n_q=snode_q
        else:
            # クエリNBのノードsnode_qの子ノードのリストを得る
            next_q_list.extend(list(self.QueryGraph.successors(snode_q)))
            if len(next_db_list) < len(next_q_list):
                return ret_list, matched

            for n in self.QueryGraph.successors(snode_q):
                if n in new_visited:
                    continue
                n_q=n
                break
        #if len(next_q_list)==0:
        if n_q is None:
            return ret_list, new_matched

        f=True
        if f:
        #for n_q in next_q_list:
            #if n_q in self.q_visited:
                #continue
            #self.q_visited.append(n_q)

            #print(new_visited)
            flg_not_found=True
            #if n_q in new_visited:
            #    continue
            for n_db in next_db_list:
                if (n_q, n_db) in new_matched:
                    continue

                if self.attr_of_q_node_type[snode_q]=="AnyWildcard":
                    pass
                elif self.attr_of_q_node_type[n_q]==self.attr_of_db_node_type[n_db]:
                    pass
                else:
                    continue

                flg_not_found=False

                flg_exist_any_n_q=False
                remaining_node_list=[]
                for n in next_q_list:
                    if n is n_q:
                        continue
                    if self.attr_of_q_node_type[n]=="AnyWildcard":
                        continue
                    if n not in new_visited:
                        remaining_node_list.append(n)
                        flg_exist_any_n_q=True

                if flg_exist_any_n_q:
                    if self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                        logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[n_q]}, {self.attr_of_db_node_type[n_db]}")
                    
                    new_new_visited=list(set(new_visited+[n_q]))
                    new_new_matched=list(set(new_matched+[(n_q, n_db)]))

                    current_score2=current_score+self.calc_rel_with_timecount(n_q, n_db)
                    if self.flg_prune_under_sim(new_visited, current_score2):
                        continue
                    current_score=current_score2

                    ret_list, matched2,=self.rec_new4_weaving(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name, current_score) 
                    ret_list, new_matched2=self.rec_new4_weaving(ret_list, n_q, n_db, new_new_visited, matched2, nb_name, current_score) 
                else:
                    ret_list, new_matched2=self.rec_new4_weaving(ret_list, n_q, n_db, new_visited, new_matched, nb_name, current_score)
                    #break #1つだけマッチすれば良い時
            if flg_not_found:
                #logging.info("no matched subgraph.")
                return [], []
            #break
        return ret_list, new_matched

    # *** 旧バージョン 提案手法  メインで使用(一括でサブグラフマッチング) エラーあり ***
    def rec_new4_weaving(self, ret_list, snode_q, snode_db, visited, matched, nb_name, current_score):
        """
        提案手法
        self.Gを利用して一括でサブグラフマッチング．(先にデータベースから一括でロード)
        クエリグラフはself.QueryGraph
        閉路を持つクエリでは誤ったサブグラフが検出される可能性あり．

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

        # クエリのノードsnode_qの子ノードのリストを得る
        
        if self.attr_of_db_node_type[snode_db] != self.attr_of_q_node_type[snode_q]:
            logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[snode_q]}, {self.attr_of_db_node_type[snode_db]}")
        
        # 類似度下限にかかっているかをチェック(下限より低ければPruning)
        current_score+=self.calc_rel_with_timecount(snode_q, snode_db)
        new_visited=list(set(visited+[(snode_q)]))
        if self.flg_running_faster["flg_prune_under_sim"] and self.flg_prune_under_sim(new_visited, current_score):
            return ret_list, matched, visited
        new_matched=matched+[(snode_q,snode_db)]

        # クエリの全てのノードに訪れた場合，終了
        if len(new_visited) == len(self.QueryGraph.nodes): # クエリの全てのノードに訪れたかどうかの判定
            new_matched=list(set(new_matched))
            new_matched.sort()
            if new_matched not in ret_list:
                ret_list.append(new_matched)
                if nb_name not in self.nb_score or self.nb_score[nb_name] < current_score:
                    self.nb_score[nb_name]=current_score
                #print(new_matched)
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

            for n in self.QueryGraph.successors(snode_q):
                if n in new_visited:
                    continue
                n_q=n
                break
        #if len(next_q_list)==0:
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

            flg_not_found=False

            #flg_exist_any_n_q=False
            #remaining_node_list=[]
            #for n in next_q_list:
            #    if n is n_q:
            #        continue
            #    if self.attr_of_q_node_type[n]=="AnyWildcard":
            #        continue
            #    if n not in new_visited:
            #        remaining_node_list.append(n)
            #        flg_exist_any_n_q=True

            #if flg_exist_any_n_q:
            if len(new_visited) != len(self.QueryGraph.nodes):

                if self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[n_q]}, {self.attr_of_db_node_type[n_db]}")
                    
                new_new_visited=list(set(new_visited+[n_q]))
                new_new_matched=list(set(new_matched+[(n_q, n_db)]))

                current_score2=current_score+self.calc_rel_with_timecount(n_q, n_db)
                if self.flg_running_faster["flg_prune_under_sim"] and self.flg_prune_under_sim(new_visited, current_score2):
                    continue
                current_score=current_score2

                ret_list, matched2, visited2 =self.rec_new4_weaving(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name, current_score) 
                ret_list, matched2, visited2 =self.rec_new4_weaving(ret_list, n_q, n_db, visited2, matched2, nb_name, current_score) 
            else:
                ret_list, matched2, visited2 =self.rec_new4_weaving(ret_list, n_q, n_db, new_visited, new_matched, nb_name, current_score)

        if flg_not_found:
            return ret_list, matched, visited
        return ret_list, matched2, visited2

    
    # *** 旧バージョン 提案手法 ***
    def rec_new4_weaving2(self, ret_list, snode_q, snode_db, visited, matched, nb_name, current_score):
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

        # 類似度下限にかかっているかをチェック(下限より低ければPruning)
        current_score2= current_score+self.calc_rel_with_timecount(snode_q, snode_db)
        new_visited=list(set(visited+[(snode_q)]))
        if self.flg_running_faster["flg_prune_under_sim"] and self.flg_prune_under_sim(new_visited, current_score2):
            return ret_list, matched, visited, current_score
        new_matched=matched+[(snode_q,snode_db)]

        # クエリの全てのノードに訪れた場合，終了
        if len(new_visited) == len(self.QueryGraph.nodes): # クエリの全てのノードに訪れたかどうかの判定
            new_matched=list(set(new_matched))
            new_matched.sort()
            if new_matched not in ret_list:
                ret_list.append(new_matched)
                if nb_name not in self.nb_score or self.nb_score[nb_name] < current_score2:
                    self.nb_score[nb_name]=current_score2
                self.flg_detection=True
                self.ans_list[nb_name].append(new_matched)
                self.detected_count+=1
            else:
                pass
                # ここの条件になる場合は存在しないはず -> 修正に伴って検出されるようになった．
                #logging.info("*************detected duplicate.*****************")
                #logging.info(new_matched)
            return ret_list, new_matched, new_visited, current_score2


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
                return ret_list, matched, visited, current_score
            for n in list(self.QueryGraph.successors(snode_q)):
                if n in new_visited:
                    continue
                n_q=n
                break
        if n_q is None:
            return ret_list, new_matched, new_visited, current_score2


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

            flg_not_found=False

            flg_exist_any_n_q=False
            remaining_node_list=[]
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
                
                #current_score3=current_score2+self.calc_rel_with_timecount(n_q, n_db)
                #if self.flg_running_faster["flg_prune_under_sim"] and self.flg_prune_under_sim(new_visited, current_score3):
                #    continue

                ret_list, matched2, visited2, current_score3 =self.rec_new4_weaving2(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name, current_score2) 
                ret_list, matched2, visited2, current_score3 =self.rec_new4_weaving2(ret_list, n_q, n_db, visited2, matched2, nb_name, current_score3) 
            else:
                ret_list, matched2, visited2, current_score3 =self.rec_new4_weaving2(ret_list, n_q, n_db, new_visited, new_matched, nb_name, current_score2)
        if flg_not_found:
            return ret_list, matched, visited, current_score
        return ret_list, matched2, visited2, current_score3
    
    # *** 提案手法 ***
    def rec_new4_weaving3(self, ret_list, snode_q, snode_db, visited, matched, nb_name, current_score):
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

        
        #if (self.attr_of_q_node_type[snode_q] not in ["OneWildcard", "AnyWildcard"]) and self.attr_of_db_node_type[snode_db] != self.attr_of_q_node_type[snode_q]:
        #    logging.info(f"error1: not match node type of workflow. {self.attr_of_q_node_type[snode_q]}, {self.attr_of_db_node_type[snode_db]}")

        # 類似度下限にかかっているかをチェック(下限より低ければPruning)
        new_visited=list(set(visited+[(snode_q)]))
        new_matched=matched+[(snode_q,snode_db)]

        # クエリの全てのノードに訪れた場合，終了
        if len(new_visited) == len(self.QueryGraph.nodes): # クエリの全てのノードに訪れたかどうかの判定
            new_matched=list(set(new_matched))
            new_matched.sort()
            if new_matched not in ret_list:
                ret_list.append(new_matched)
                if nb_name not in self.nb_score or self.nb_score[nb_name] < current_score:
                    self.nb_score[nb_name]=current_score
                self.flg_detection=True
                self.ans_list[nb_name].append(new_matched)
                self.detected_count+=1
            else:
                pass
                # ここの条件になる場合は存在しないはず -> 修正に伴って検出されるようになった．
                logging.info("*************detected duplicate.*****************")
                logging.info(new_matched)
            return ret_list, new_matched, new_visited, current_score


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
                return ret_list, matched, visited, current_score
            for n in list(self.QueryGraph.successors(snode_q)):
                if n in new_visited:
                    continue
                n_q=n
                break
        if n_q is None:
            return ret_list, new_matched, new_visited, current_score


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


            flg_exist_any_n_q=False
            #remaining_node_list=[]
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
                if (self.attr_of_q_node_type[n_q] not in ["OneWildcard", "AnyWildcard"]) and self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    logging.info(f"error2: not match node type of workflow. {self.attr_of_q_node_type[n_q]}, {self.attr_of_db_node_type[n_db]}")
                #new_new_matched=list(set(new_matched+[(n_q, n_db)]))
                
                current_score2=current_score+self.calc_rel_with_timecount(n_q, n_db)
                if self.flg_running_faster["flg_prune_under_sim"] and self.flg_prune_under_sim(list(set(new_visited+[n_q])), current_score2):
                    continue
                flg_not_found=False
                new_new_visited=list(set(new_visited+[n_q]))
                new_new_matched=new_matched+[(n_q, n_db)]

                ret_list, matched2, visited2, current_score3 =self.rec_new4_weaving3(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name, current_score2) 
                ret_list, matched2, visited2, current_score3 =self.rec_new4_weaving3(ret_list, n_q, n_db, visited2, matched2, nb_name, current_score3) 
            else:
                current_score2=current_score+self.calc_rel_with_timecount(n_q, n_db)
                if self.flg_running_faster["flg_prune_under_sim"] and self.flg_prune_under_sim(list(set(new_visited+[n_q])), current_score2):
                    continue
                flg_not_found=False
                ret_list, matched2, visited2, current_score3 =self.rec_new4_weaving3(ret_list, n_q, n_db, new_visited, new_matched, nb_name, current_score2)
        if flg_not_found:
            return ret_list, matched, visited, current_score
        #return ret_list, new_matched, new_visited, current_score
        return ret_list, matched2, visited2, current_score3
    
    # 提案手法が動作が重いので確認用 (使用終わったので削除可能)
    def rec_new4_weaving3_chk(self, ret_list, snode_q, snode_db, visited, matched, nb_name, current_score):
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
                if nb_name not in self.nb_score or self.nb_score[nb_name] < current_score:
                    self.nb_score[nb_name]=current_score
                self.flg_detection=True
                self.ans_list[nb_name].append(new_matched)
                self.detected_count+=1
            else:
                pass
                # ここの条件になる場合は存在しないはず -> 修正に伴って検出されるようになった．
                #logging.info("*************detected duplicate.*****************")
                #logging.info(new_matched)
            return ret_list, new_matched, new_visited, current_score


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
                return ret_list, matched, visited, current_score
            for n in list(self.QueryGraph.successors(snode_q)):
                if n in new_visited:
                    continue
                n_q=n
                break
        if n_q is None:
            return ret_list, new_matched, new_visited, current_score


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


            flg_exist_any_n_q=False
            for n in next_q_list:
                if n is n_q:
                    continue
                if self.attr_of_q_node_type[n]=="AnyWildcard":
                    continue
                if n not in new_visited:
                    flg_exist_any_n_q=True
            if flg_exist_any_n_q:
                if (self.attr_of_q_node_type[n_q] not in ["OneWildcard", "AnyWildcard"]) and self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    logging.info(f"error2: not match node type of workflow. {self.attr_of_q_node_type[n_q]}, {self.attr_of_db_node_type[n_db]}")
                current_score2=current_score+self.calc_rel_with_timecount(n_q, n_db)
                flg_not_found=False
                new_new_visited=list(set(new_visited+[n_q]))
                new_new_matched=new_matched+[(n_q, n_db)]
                ret_list, matched2, visited2, current_score3=self.rec_new4_weaving3_chk(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name, current_score2) 
                ret_list, matched2, visited2, current_score3=self.rec_new4_weaving3_chk(ret_list, n_q, n_db, visited2, matched2, nb_name, current_score3) 
            else:
                current_score2=current_score+self.calc_rel_with_timecount(n_q, n_db)
                flg_not_found=False
                ret_list, matched2, visited2, current_score3 =self.rec_new4_weaving3_chk(ret_list, n_q, n_db, new_visited, new_matched, nb_name, current_score2)
        if flg_not_found:
            return ret_list, matched, visited, current_score
        return ret_list, matched2, visited2, current_score3

    # *** メインで使用(一括でサブグラフマッチング) ***
    def rec_new4_weaving4(self, ret_list, snode_q, snode_db, visited, matched, nb_name):
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
                current_score=self.calc_one_score(new_matched)
                if nb_name not in self.nb_score or self.nb_score[nb_name] < current_score:
                    self.nb_score[nb_name]=current_score
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
                ret_list, matched2, visited2=self.rec_new4_weaving4(ret_list, snode_q, snode_db, new_new_visited, new_new_matched, nb_name) 
                ret_list, matched2, visited2=self.rec_new4_weaving4(ret_list, n_q, n_db, visited2, matched2, nb_name) 
            else:
                flg_not_found=False
                ret_list, matched2, visited2 =self.rec_new4_weaving4(ret_list, n_q, n_db, new_visited, new_matched, nb_name)
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


    def calc_rel_with_timecount(self, n1_name, n2_name):
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        if not self.flg_running_faster["flg_caching"]:
            self.calculated_sim={}
        return ret

    def calc_rel_with_timecount_old(self, n1_name, n2_name):
        class_code_relatedness=self.class_code_relatedness
        if self.attr_of_db_node_type[n2_name] == "Cell" and self.attr_of_q_node_type[n1_name] == "Cell": #type: cell
            start_time = timeit.default_timer()
            if self.w_c==0:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Cell"]+=end_time-start_time
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Cell"]+=end_time-start_time
                return self.calculated_sim[(n1_name, n2_name)] * self.w_c
            #n1_real_cell_id=self.attr_of_q_real_cell_id[n1_name]
            n2_real_cell_id=self.attr_of_db_real_cell_id[n2_name]
            #print(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id)
            #sim = self.calc_cell_rel(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id, ver="jaccard_similarity_coefficient")
            code_A=self.query_cell_code[n1_name]
            cleaned_nb_name_B=self.attr_of_db_nb_name[n2_name]
            code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
            code_B=code_table_B[code_table_B["cell_id"]==n2_real_cell_id]["cell_code"].values
            code_B="".join(list(code_B))

            sim=class_code_relatedness.calc_code_rel_by_jaccard_index(code_A, code_B)
            self.calculated_sim[(n1_name, n2_name)]=sim
            end_time = timeit.default_timer()
            self.each_calc_time_sum["Cell"]+=end_time-start_time
            return  sim * self.w_c
        elif self.attr_of_db_node_type[n2_name] == "Var" and self.attr_of_q_node_type[n1_name] == "Var": #type: var
            start_time = timeit.default_timer()
            self.calc_v_count+=1
            if self.w_v==0:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Var"]+=end_time-start_time
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Var"]+=end_time-start_time
                return self.calculated_sim[(n1_name, n2_name)] * self.w_v
            #tableA=self.fetch_var_table(n1_name)
            if n1_name not in self.query_table:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Var"]+=end_time-start_time
                return 0.0
            tableA=self.query_table[n1_name]
            #if tableA is None:
            #    return 0.0
            tableB=self.fetch_var_table(n2_name)
            if tableB is None:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Var"]+=end_time-start_time
                return 0.0
            sim = self.calc_table_rel(tableA, tableB)
            self.calculated_sim[(n1_name, n2_name)]=sim
            end_time = timeit.default_timer()
            self.each_calc_time_sum["Var"]+=end_time-start_time
            return sim * self.w_v
        elif self.attr_of_db_node_type[n2_name] == "Display_data" and self.attr_of_q_node_type[n1_name] == "Display_data": #type: Display_data
            start_time = timeit.default_timer()
            if self.w_d==0:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Display_data"]+=end_time-start_time
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Display_data"]+=end_time-start_time
                return self.calculated_sim[(n1_name, n2_name)]
            if n1_name not in self.attr_of_q_display_type:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Display_data"]+=end_time-start_time
                return 0.0
            if n2_name not in self.attr_of_db_display_type:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Display_data"]+=end_time-start_time
                return 0.0
            if self.attr_of_db_display_type[n2_name] == self.attr_of_q_display_type[n1_name]:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Display_data"]+=end_time-start_time
                return self.w_d
            else:
                end_time = timeit.default_timer()
                self.each_calc_time_sum["Display_data"]+=end_time-start_time
                return 0.0
        else:
            #logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[n1_name]}, {self.attr_of_db_node_type[n2_name]}")
            return 0.0

    def calc_rel_c_l_o_with_timecount(self, n1_name, n2_name):
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel_c_o(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        if not self.flg_running_faster["flg_caching"]:
            self.calculated_sim={}
        return ret

    def calc_rel_o_with_timecount(self, n1_name, n2_name):
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel_o(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        if not self.flg_running_faster["flg_caching"]:
            self.calculated_sim={}
        return ret

    def calc_rel_v_with_timecount(self, n1_name, n2_name):
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel_v(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        if not self.flg_running_faster["flg_caching"]:
            self.calculated_sim={}
        return ret

    def calc_rel_c_with_timecount_2(self, n1_name, n2_name, remain_c_count):
        calc_start_time = timeit.default_timer()
        ret1, ret2=self.calc_rel_c_2(n1_name, n2_name, remain_c_count)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        if not self.flg_running_faster["flg_caching"]:
            self.calculated_sim={}
        return ret1, ret2

    def calc_rel_v_with_timecount_2(self, n1_name, n2_name, remain_v_count):
        calc_start_time = timeit.default_timer()
        ret1, ret2=self.calc_rel_v_2(n1_name, n2_name, remain_v_count)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        if not self.flg_running_faster["flg_caching"]:
            self.calculated_sim={}
        return ret1, ret2

    def calc_rel_v_approximately_with_timecount(self, n1_name, n2_name):
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel_v_approximately(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        return ret

    def calc_rel(self, n1_name, n2_name):
        class_code_relatedness=self.class_code_relatedness
        if self.attr_of_db_node_type[n2_name] == "Cell" and self.attr_of_q_node_type[n1_name] == "Cell": #type: cell
            if self.w_c==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)] * self.w_c
            #n1_real_cell_id=self.attr_of_q_real_cell_id[n1_name]
            n2_real_cell_id=self.attr_of_db_real_cell_id[n2_name]
            #print(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id)
            #sim = self.calc_cell_rel(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id, ver="jaccard_similarity_coefficient")
            code_A=self.query_cell_code[n1_name]
            cleaned_nb_name_B=self.attr_of_db_nb_name[n2_name]
            code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
            code_B=code_table_B[code_table_B["cell_id"]==n2_real_cell_id]["cell_code"].values
            code_B="".join(list(code_B))

            sim=class_code_relatedness.calc_code_rel_by_jaccard_index(code_A, code_B)
            self.calculated_sim[(n1_name, n2_name)]=sim
            return  sim * self.w_c
        elif self.attr_of_db_node_type[n2_name] == "Var" and self.attr_of_q_node_type[n1_name] == "Var": #type: var
            self.calc_v_count+=1
            if self.w_v==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)] * self.w_v
            #tableA=self.fetch_var_table(n1_name)

            start_time_load = timeit.default_timer()
            if n1_name not in self.query_table:
                return 0.0
            tableB=self.fetch_var_table(n2_name)
            if tableB is None:
                return 0.0
            if self.flg_running_faster["flg_cache_query_table"]:
                end_time_load = timeit.default_timer()
                self.calc_v_microbenchmark["load"]+=end_time_load-start_time_load
                sim = self.calc_table_rel_between_query_and_db(n1_name, tableB)
            else:
                tableA=self.query_table[n1_name]
                end_time_load = timeit.default_timer()
                self.calc_v_microbenchmark["load"]+=end_time_load-start_time_load
                #if tableA is None:
                #    return 0.0
                sim = self.calc_table_rel(tableA, tableB)
            self.calculated_sim[(n1_name, n2_name)]=sim
            return sim * self.w_v
        elif self.attr_of_db_node_type[n2_name] == "Display_data" and self.attr_of_q_node_type[n1_name] == "Display_data": #type: Display_data
            if self.w_d==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)]
            if n1_name not in self.attr_of_q_display_type:
                return 0.0
            if n2_name not in self.attr_of_db_display_type:
                return 0.0
            if self.attr_of_db_display_type[n2_name] == self.attr_of_q_display_type[n1_name]:
                return self.w_d
            else:
                return 0.0
        else:
            #logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[n1_name]}, {self.attr_of_db_node_type[n2_name]}")
            return 0.0


    def calc_rel_c_o(self, n1_name, n2_name):
        class_code_relatedness=self.class_code_relatedness
        if self.attr_of_db_node_type[n2_name] == "Cell" and self.attr_of_q_node_type[n1_name] == "Cell": #type: cell
            if self.w_c==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)] * self.w_c
            #n1_real_cell_id=self.attr_of_q_real_cell_id[n1_name]
            n2_real_cell_id=self.attr_of_db_real_cell_id[n2_name]
            #print(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id)
            #sim = self.calc_cell_rel(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id, ver="jaccard_similarity_coefficient")
            code_A=self.query_cell_code[n1_name]
            cleaned_nb_name_B=self.attr_of_db_nb_name[n2_name]
            code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
            code_B=code_table_B[code_table_B["cell_id"]==n2_real_cell_id]["cell_code"].values
            code_B="".join(list(code_B))

            sim=class_code_relatedness.calc_code_rel_by_jaccard_index(code_A, code_B)
            self.calculated_sim[(n1_name, n2_name)]=sim
            return  sim * self.w_c
        elif self.attr_of_db_node_type[n2_name] == "Var" and self.attr_of_q_node_type[n1_name] == "Var": #type: var
            return 0
        elif self.attr_of_db_node_type[n2_name] == "Display_data" and self.attr_of_q_node_type[n1_name] == "Display_data": #type: Display_data
            if self.w_d==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)]
            if n1_name not in self.attr_of_q_display_type:
                return 0.0
            if n2_name not in self.attr_of_db_display_type:
                return 0.0
            if self.attr_of_db_display_type[n2_name] == self.attr_of_q_display_type[n1_name]:
                return self.w_d
            else:
                return 0.0
        else:
            #logging.info(f"error: not match node type of workflow. {self.attr_of_q_node_type[n1_name]}, {self.attr_of_db_node_type[n2_name]}")
            return 0.0

    def calc_rel_c_2(self, n1_name, n2_name, remain_c_count):
        class_code_relatedness=self.class_code_relatedness
        if self.attr_of_db_node_type[n2_name] == "Cell" and self.attr_of_q_node_type[n1_name] == "Cell": #type: cell
            remain_c_count-=1
            if self.w_c==0:
                return 0, remain_c_count
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)] * self.w_c, remain_c_count
            #n1_real_cell_id=self.attr_of_q_real_cell_id[n1_name]
            n2_real_cell_id=self.attr_of_db_real_cell_id[n2_name]
            #print(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id)
            #sim = self.calc_cell_rel(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id, ver="jaccard_similarity_coefficient")
            code_A=self.query_cell_code[n1_name]
            cleaned_nb_name_B=self.attr_of_db_nb_name[n2_name]
            code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
            code_B=code_table_B[code_table_B["cell_id"]==n2_real_cell_id]["cell_code"].values
            code_B="".join(list(code_B))

            sim=class_code_relatedness.calc_code_rel_by_jaccard_index(code_A, code_B)
            self.calculated_sim[(n1_name, n2_name)]=sim
            return sim * self.w_c, remain_c_count
        else:
            return 0.0, remain_c_count

    def calc_rel_o(self, n1_name, n2_name):
        class_code_relatedness=self.class_code_relatedness
        if self.attr_of_db_node_type[n2_name] == "Display_data" and self.attr_of_q_node_type[n1_name] == "Display_data": #type: Display_data
            if self.w_d==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)]
            if n1_name not in self.attr_of_q_display_type:
                return 0.0
            if n2_name not in self.attr_of_db_display_type:
                return 0.0
            if self.attr_of_db_display_type[n2_name] == self.attr_of_q_display_type[n1_name]:
                return self.w_d
            else:
                return 0.0
        else:
            return 0.0


    def calc_rel_v(self, n1_name, n2_name):
        if self.attr_of_db_node_type[n2_name] == "Var" and self.attr_of_q_node_type[n1_name] == "Var": #type: var
            self.calc_v_count+=1
            if self.w_v==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)] * self.w_v
            start_time_load = timeit.default_timer()
            if n1_name not in self.query_table:
                return 0.0
            tableB=self.fetch_var_table(n2_name)
            if tableB is None:
                return 0.0
            if self.flg_running_faster["flg_cache_query_table"]:
                end_time_load = timeit.default_timer()
                self.calc_v_microbenchmark["load"]+=end_time_load-start_time_load
                sim = self.calc_table_rel_between_query_and_db(n1_name, tableB)
            else:
                tableA=self.query_table[n1_name]
                end_time_load = timeit.default_timer()
                self.calc_v_microbenchmark["load"]+=end_time_load-start_time_load
                sim = self.calc_table_rel(tableA, tableB)
            self.calculated_sim[(n1_name, n2_name)]=sim
            return sim * self.w_v
        else:
            return 0.0

    def calc_rel_v_2(self, n1_name, n2_name, remain_v_count):
        class_code_relatedness=self.class_code_relatedness
        if self.attr_of_db_node_type[n2_name] == "Var" and self.attr_of_q_node_type[n1_name] == "Var": #type: var
            self.calc_v_count+=1
            remain_v_count-=1
            if self.w_v==0:
                return 0, remain_v_count
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)] * self.w_v, remain_v_count

            start_time_load = timeit.default_timer()
            if n1_name not in self.query_table:
                return 0.0, remain_v_count
            tableB=self.fetch_var_table(n2_name)
            if tableB is None:
                return 0.0, remain_v_count
            if self.flg_running_faster["flg_cache_query_table"]:
                end_time_load = timeit.default_timer()
                self.calc_v_microbenchmark["load"]+=end_time_load-start_time_load
                sim = self.calc_table_rel_between_query_and_db(n1_name, tableB)
            else:
                tableA=self.query_table[n1_name]
                end_time_load = timeit.default_timer()
                self.calc_v_microbenchmark["load"]+=end_time_load-start_time_load
                sim = self.calc_table_rel(tableA, tableB)
            self.calculated_sim[(n1_name, n2_name)]=sim
            return sim * self.w_v, remain_v_count
        else:
            return 0.0, remain_v_count

    def calc_rel_v_approximately(self, n1_name, n2_name):
        if self.w_v==0:
            return 0
        if self.attr_of_db_node_type[n2_name] == "Var" and self.attr_of_q_node_type[n1_name] == "Var": #type: var
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
    # 使用
    def calc_rel_between_db_node_with_timecount(self, n1_name, n2_name):
        calc_start_time = timeit.default_timer()
        ret=self.calc_rel_between_db_node(n1_name, n2_name)
        calc_end_time = timeit.default_timer()
        self.calc_time_sum+=calc_end_time-calc_start_time
        self.each_calc_time_sum[self.attr_of_db_node_type[n2_name]]+=calc_end_time-calc_start_time
        if not self.flg_running_faster["flg_caching"]:
            self.calculated_sim={}
        return ret

    # 使用
    def calc_rel_between_db_node(self, n1_name, n2_name):
        class_code_relatedness=self.class_code_relatedness
        if self.attr_of_db_node_type[n2_name] == "Cell" and self.attr_of_db_node_type[n1_name] == "Cell": #type: cell
            if self.w_c==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)] * self.w_c
            n1_real_cell_id=self.attr_of_db_real_cell_id[n1_name]
            n2_real_cell_id=self.attr_of_db_real_cell_id[n2_name]
            #print(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id)
            #sim = self.calc_cell_rel(nb_name_A, nb_name_B, n1_real_cell_id, n2_real_cell_id, ver="jaccard_similarity_coefficient")
            cleaned_nb_name_A=self.attr_of_db_nb_name[n1_name]
            code_table_A=self.fetch_source_code_table(cleaned_nb_name_A)
            code_A=code_table_A[code_table_A["cell_id"]==n1_real_cell_id]["cell_code"].values
            code_A="".join(list(code_A))

            cleaned_nb_name_B=self.attr_of_db_nb_name[n2_name]
            code_table_B=self.fetch_source_code_table(cleaned_nb_name_B)
            code_B=code_table_B[code_table_B["cell_id"]==n2_real_cell_id]["cell_code"].values
            code_B="".join(list(code_B))

            sim=class_code_relatedness.calc_code_rel_by_jaccard_index(code_A, code_B)
            self.calculated_sim[(n1_name, n2_name)]=sim
            return  sim * self.w_c
        elif self.attr_of_db_node_type[n2_name] == "Var" and self.attr_of_db_node_type[n1_name] == "Var": #type: var
            self.calc_v_count+=1
            if self.w_v==0:
                return 0
            if (n1_name, n2_name) in self.calculated_sim:
                return self.calculated_sim[(n1_name, n2_name)] * self.w_v
            tableA=self.fetch_var_table(n1_name)
            if tableA is None:
                return 0.0
            tableB=self.fetch_var_table(n2_name)
            if tableB is None:
                return 0.0
            sim = self.calc_table_rel(tableA, tableB)
            self.calculated_sim[(n1_name, n2_name)]=sim
            return sim * self.w_v
        elif self.attr_of_db_node_type[n2_name] == "Display_data" and self.attr_of_db_node_type[n1_name] == "Display_data": #type: Display_data
            if self.w_d==0:
                return 0
            if n1_name not in self.attr_of_db_display_type:
                return 0.0
            if n2_name not in self.attr_of_db_display_type:
                return 0.0
            if self.attr_of_db_display_type[n2_name] == self.attr_of_db_display_type[n1_name]:
                return self.w_d
            else:
                return 0.0
        else:
            logging.info(f"error: not match node type of workflow. {self.attr_of_db_display_type[n1_name]}, {self.attr_of_db_node_type[n2_name]}")
            return 0.0

    # 使用
    def calc_nb_score3(self, w_c=None, w_v=None, w_l=None, w_d=None):
        """
        スコア計算部分の時間計測あり
        self.ans_listの集合 --- ノード名(str)

        Args:
            w_c: セル類似度をNBスコアにする際の重み
            w_v: テーブル類似度をNBスコアにする際の重み
        """

        self.init_each_calc_time_sum()
        for w in [w_c, w_v, w_l, w_d]:
            if not w is None:
                self.set_each_w(w_c, w_v, w_l, w_d)
                break

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
    def calc_nb_score3_2(self, w_c=None, w_v=None, w_l=None, w_d=None):
        """
        スコア計算部分の時間計測あり
        self.ans_listの集合 --- ノード名(str)

        Args:
            w_c: セル類似度をNBスコアにする際の重み
            w_v: テーブル類似度をNBスコアにする際の重み
        """
        self.set_flg_running_faster(False, False, False, False)
        self.init_each_calc_time_sum()
        for w in [w_c, w_v, w_l, w_d]:
            if not w is None:
                self.set_each_w(w_c, w_v, w_l, w_d)
                break

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
    def calc_nb_score_for_new_proposal_method_2_3(self, w_c=None, w_v=None, w_l=None, w_d=None):
        """
        スコア計算部分の時間計測あり
        self.ans_listの集合 --- ノード名(str)
        データ関連度の重みがゼロの時はもっと高速な別の関数に切り替える．
        この関数ではデータ関連度を1つ計算するたびに類似度下限値との比較を行う．
        類似度下限値との比較の関数もver2_2より一部改良している．(計算回数を減らす)
        + flg_running_fasterの比較回数を最小限にした．
        """
        self.init_each_calc_time_sum()
        for w in [w_c, w_v, w_l, w_d]:
            if not w is None:
                self.set_each_w(w_c, w_v, w_l, w_d)
                break
        if not self.flg_running_faster["flg_prune_under_sim"]:
            return self.calc_nb_score3_2()
        if self.w_v==0:
            return self.calc_nb_score_for_new_proposal_method_2_3_wv0()

        self.nb_score={}
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
                    current_score+=self.calc_rel_c_l_o_with_timecount(n1_name, n2_name)
                    if self.flg_running_faster["flg_calc_data_sim_approximately"]:
                        v_score_appr+=self.calc_rel_v_approximately(n1_name, n2_name)
            
                #logging.info("score is ", current_score)
                #subgraph_score[subgraph]=current_score
                subgraph_id[graph_id]=subgraph
                subgraph_score[graph_id]=current_score
                subgraph_score_appr[graph_id]=current_score + v_score_appr
                subgraph_nb_name[graph_id]=nb_name


        if self.flg_running_faster["flg_optimize_calc_order"]:
            if self.flg_running_faster["flg_calc_data_sim_approximately"]:
                sorted_subgraph=sorted(subgraph_score_appr.items(), key=lambda d: d[1], reverse=True)
            else:
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
                v_score, remain_v_count=self.calc_rel_v_with_timecount_2(n1_name, n2_name, remain_v_count)
                current_score+=v_score
                if remain_v_count==0 or self.flg_prune_under_sim_for_new_proposal_method(remain_v_count, current_score, max(k_score, self.nb_score[nb_name])):
                    break

            #logging.info("score is ", current_score)
            if self.nb_score[nb_name] < current_score:
                self.nb_score[nb_name]=current_score
            count+=1 #あとで消す
        nb_count+=1 #あとで消す

        return count, nb_count #あとで消す

    # 使用 現状最速
    def calc_nb_score_for_new_proposal_method_2_3_wv0(self, w_c=None, w_v=None, w_l=None, w_d=None):
        """
        w_v=0のとき
        """
        self.nb_score={}
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
                c_score, remain_c_count=self.calc_rel_c_with_timecount_2(n1_name, n2_name, remain_c_count)
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
    def calc_lib_score(self, lib_list1, lib_list2):
        """
        ライブラリ類似度を計算する．

        Args:
            lib_list1 (list[str]): list of name of libraries using in the notebook
            lib_list2 (list[str]): list of name of libraries using in the notebook

        Returns:
            float-like: similarity score between given libraries lists.
        """
        calc_lib_start_time = timeit.default_timer()
        lib_score = self.jaccard_similarity_coefficient(lib_list1, lib_list2)
        calc_lib_end_time = timeit.default_timer()
        self.each_calc_time_sum["Library"]+=calc_lib_end_time-calc_lib_start_time
        return lib_score

    def top_k_nb(self, k):
        #for nb_name in self.ans_list:
        #    for subgraph in self.ans_list[nb_name]:
        #        for n_tuple in subgraph:
        #            if type(n_tuple[0]) == type("str"):
        #                self.calc_nb_score2()
        #            else:
        #                self.calc_nb_score()
        #            break
        #        break
        #    break

        #print(self.nb_score)
            
        sorted_nb_score = sorted(self.nb_score.items(), key=lambda x:x[1], reverse=True)
        self.k = min(len(sorted_nb_score), k)
        #ret=[]
        #for i in range(k):
            #ret.append(sorted_nb_score[i])
        ret=sorted_nb_score[:self.k]
        self.top_k_score=ret
        return ret
         
    def top_k_nb_with_display_data(self, k):
        self.top_k_score_d={}
        self.top_k_score_l=[]
        for nb_name in self.ans_list:
            for subgraph in self.ans_list[nb_name]:
                for n_tuple in subgraph:
                    if type(n_tuple[0]) == type("str"):
                        self.calc_nb_score2()
                    else:
                        self.calc_nb_score()
                    break
                break
            break

        print(self.nb_score)
            
        sorted_nb_score = sorted(self.nb_score.items(), key=lambda x:x[1])
        self.k = min(len(sorted_nb_score), k)
        #self.top_k_score=sorted_nb_score[:self.k]

        count=0
        for nb_name in sorted_nb_score:
            flg_lib_matched=True
            display_db=self.fetch_multiset_of_display_type_from_db(nb_name)
            for t in self.query_display_type:
                if not t in display_db:
                    flg_lib_matched=False
                    break
            if flg_lib_matched:
                self.top_k_score_d[nb_name]=sorted_nb_score[nb_name]
                self.top_k_score_l.append((nb_name, sorted_nb_score[nb_name]))  
        print(self.top_k_score)       

    def fetch_source_code_table(self, cleaned_nb_name):
        #return 0.5
        if cleaned_nb_name in self.cell_source_code:
            return self.cell_source_code[cleaned_nb_name]
        
        with self.postgres_eng.connect() as conn:
            try:
                code_table = pd.read_sql_table(f"cellcont_{cleaned_nb_name}", conn, schema=f"{config.sql.cellcode}")
                self.cell_source_code[cleaned_nb_name]=code_table
            except:
                logging.info(f"error: collecting source code from db is failed.")
        return self.cell_source_code[cleaned_nb_name]
        
    def fetch_var_table(self, table_name):
        """
        Args:
            table_name (str): f'rtable{セル番号}_{変数名}_{NB名}'の文字列．
        
        Returns:
            DataFrame
        """
        with self.postgres_eng.connect() as conn:
            try:
                var_table = pd.read_sql_table(f"rtable{table_name}", conn, schema=f"{config.sql.dbs}")
                return var_table
            except Exception as e:
                logging.info(f"error: collecting var table '{table_name}' from db is failed because of {e}")
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


    def calc_cell_rel(self, cleaned_nb_name_A, cleaned_nb_name_B, cell_id_A, cell_id_B, ver="jaccard_similarity_coefficient"):
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


    def bench_mark_calc_table_rel(self, limit=100):
        time2=0
        calc_count=0
        node_list=[]
        attr_list=nx.get_node_attributes(self.G, "node_type")
        for n in attr_list:
            if attr_list[n]=="Var":
                node_list.append(n)

        start_time1 = timeit.default_timer()
        for n1_name in node_list:
            if calc_count>=limit:
                break
            for n2_name in node_list:
                if calc_count>=limit:
                    break
                if n1_name==n2_name:
                    continue
                start_time2 = timeit.default_timer()
                tableA=self.fetch_var_table(n1_name)
                if tableA is None:
                    continue
                tableB=self.fetch_var_table(n2_name)
                if tableB is None:
                    continue
                sim = self.calc_table_rel(tableA, tableB)
                end_time2 = timeit.default_timer()
                time2+=end_time2-start_time2
                print(sim)
                calc_count+=1
        end_time1 = timeit.default_timer()
        time1=end_time1-start_time1
        logging.info(f"calculated num of nb: {len(self.ans_list)}")
        logging.info(f"calc time par nb: {time2/len(self.ans_list)}")
        logging.info(f"calc time par 1 set of table: {time2/calc_count}")
        return time1, time2, calc_count

    def calc_table_rel_old(self, tableA, tableB):
        """
        修正前版．（''57_pubwisegame_deogamesalesedawithplotly''などでエラー発生．）
        この関数の引数にするテーブルは関数self.fetch_var_table(table_name)で取得できる。
        """
        #group idはnb_nameで代用できる(つながっているワークフローごとに一意なので)
        logging.info("calc table sim...")

        rel_score=0.0
        matching=[]
        acol_set = {} # dict{str: list[any]}: {列名: 列名に対応する列のデータ値のうち,Null値を除いたデータ値の重複無しリスト.}
        matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名A: 列名B}}
        r_matched_pair = {} # dict{int: dict{str: str}}: 内容は{グループID: {列名B: 列名A}}

        scmaA = tableA.columns.values # scmaA (list[str]): tableAの列名リスト
        scmaB = tableB.columns.values # scmaB (list[str]): tableBの列名リスト
        for nameA in scmaA:
            if nameA == "Unnamed: 0" or "index" in nameA or nameA=="Unnamed:0":
                continue

            if nameA not in acol_set: #以前にデータ値を処理したものは取っておく
                colA = tableA[nameA][~pd.isnull(tableA[nameA])].values # ~はビット反転演算子
                acol_set[nameA] = list(set(colA)) # 重複無しリスト
            else:
                colA=acol_set[nameA]

            for nameB in scmaB:
                if nameB == "Unnamed: 0" or "index" in nameB or nameB=="Unnamed:0":
                    continue
                if tableA[nameA].dtype is not tableB[nameB].dtype: # 列nameAと列nameBのデータ型が異なる場合
                    continue

                if nameB not in acol_set:
                    colB = tableB[nameB][~pd.isnull(tableB[nameB])].values # ~はビット反転演算子
                    acol_set[nameB] = list(set(colB)) # 重複無しリスト
                else:
                    colB=acol_set[nameB]

                sim_col = self.jaccard_similarity_coefficient(colA, colB)
                matching.append((nameA, nameB, sim_col)) # list[str, str, float]: [列名A, 列名B, 列Aと列Bの類似度]

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

    def calc_table_rel(self, tableA, tableB):
        """
        一時的に _tmp 付与している
        引数で与えた2テーブルに対し，ジャカード係数を元に類似度を計算する．

        バグ修正版．
        バグ：同じ列名がtableAとtableBに含まれているときacol_setのキャッシュが原因でエラーが発生．

        Args:
            tableA (DataFrame)
            tableB (DataFrame)
        
        Returns:
            float-like: tableAとtableBの類似度．
        """
        #group idはnb_nameで代用できる(つながっているワークフローごとに一意なので)
        logging.info("calc table sim...")

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

    #def cache_query_table(self):
    #    if self.flg_running_faster["flg_cache_query_table"]:
    #        for name in self.query_table:
    #            scma = self.query_table[name].columns.values # scmaA (list[str]): tableAの列名リスト
    #            self.query_scma_dict[name]=scma

    def calc_table_rel_between_query_and_db(self, query_node_name, tableB):
        """
        引数で与えた2テーブルに対し，ジャカード係数を元に類似度を計算する．

        バグ修正版．
        バグ：同じ列名がtableAとtableBに含まれているときacol_setのキャッシュが原因でエラーが発生．

        Args:
            tableB (DataFrame)
        
        Returns:
            float-like: tableAとtableBの類似度．
        """
        #group idはnb_nameで代用できる(つながっているワークフローごとに一意なので)
        logging.info("calc table sim...")
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


    def fetch_all_library_from_db(self, cleaned_nb_name):
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
        

    def fetch_library_in_particular_nb_from_db(self, nb_name):
        if nb_name in self.library:
            return self.library[nb_name]

        logging.info("collecting library info...")
        with self.postgres_eng.connect() as conn:
            #conn = self.postgres_eng
            #conn = self.postgres_eng.connect()
            try:
                lib_table = pd.read_sql_table("nb_libraries", conn, schema=f"{config.sql.nb_info}")
                logging.info("***")
                for _, row in lib_table.iterrows():
                    print(row["nb_name"], row["libraries"])
                for _, nb_name, lib in lib_table.iterrows():
                    print(f"{_}, {nb_name}, {lib}")
                    if nb_name not in self.library:
                        self.library[nb_name]=[]
                    if lib not in self.library[nb_name]:
                        self.library[nb_name].append(lib)
                logging.info("collecting library info : completed.")
            except:
                logging.info("collecting library info : failed.")
        return self.library[nb_name]


    def fetch_multiset_of_display_type_from_db(self, cleaned_nb_name):
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

    #不使用?
    def fetch_cell_source_code_from_db(self, cleaned_nb_name):
        """
        Returns:
            dict{int-like: str}: key is cell id and value is source code in the cell.
        """
        if cleaned_nb_name in self.cell_source_code:
            return self.cell_source_code[cleaned_nb_name]
        logging.info("collecting cell source code info...")
        with self.postgres_eng.connect() as conn:
            try:
                d_table = pd.read_sql_table(f"cellcont_{cleaned_nb_name}", conn, schema=f"{config.sql.cellcode}")
                for _, row in d_table.iterrows():
                    cid=row["cell_id"]
                    src_code=row["cell_code"]
                    if cid not in self.cell_source_code:
                        self.cell_source_code[cleaned_nb_name]={}
                    if cid not in self.cell_source_code[cleaned_nb_name]:
                        self.cell_source_code[cleaned_nb_name][cid]=src_code
                logging.info("collecting source code info : completed.")
            except:
                logging.info("collecting source code info : failed.")
        return self.cell_source_code[cleaned_nb_name]


    def calc_library_sim(self, cleaned_nb_name1, cleaned_nb_name2):
        lib_list1 = self.fetch_all_library_from_db(cleaned_nb_name1)
        lib_list2 = self.fetch_all_library_from_db(cleaned_nb_name2)
        return self.calc_lib_score(lib_list1, lib_list2)

    # 提案手法で使用
    def flg_prune_under_sim(self, visited, current_score, max_c_score=1, max_v_score=1, max_d_score=1):
        """
        Args:
            max_c_score (int or float or int-like): とりうるセル類似度の最大値
            max_v_score (int or float or int-like): とりうるテーブル類似度の最大値
        """
        if len(self.nb_score) < self.k:
            return False
        
        # k番目のスコアの値をセット
        #count=0
        #k_score=0
        self.top_k_score=sorted(self.nb_score.items(), key=lambda d: d[1], reverse=True)
        #for nb_name in self.top_k_score:
        #    count+=1
        #    if count==self.k:
        #        k_score=self.top_k_score[nb_name]
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
    def flg_prune_under_sim_for_new_proposal_method_3(self, weight, remain_count, current_score, compare_score, max_score=1):
        under_limit = compare_score - weight * (remain_count * max_score) - current_score
        if under_limit>0:
            return True
        else:
            return False

    # 不使用（旧：新しい方の提案手法 関数new_proposal_method で使用 --ver.2）
    def flg_prune_under_sim_for_new_proposal_method_2(self, v_count, current_score, compare_score_list, max_v_score=1):
        max_score = self.w_v * (v_count * max_v_score) + current_score
        for compare_score in compare_score_list:
            if compare_score > max_score: # 枝刈りの時
                return True
        return False

    @staticmethod
    def jaccard_similarity_coefficient(colA, colB):
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
        if min(len(colA), len(colB)) == 0:
            return 0
        colA = np.array(colA) #numpyの型に変換?
        # 疑問: colBは変換しなくて良いのか？
        colB = np.array(colB) #numpyの型に変換?
        union = len(np.union1d(colA, colB))
        inter = len(np.intersect1d(colA, colB))
        return float(inter) / float(union)

    @staticmethod
    def row_similarity(colA, colB):
        # search/search_tables.pyからコピー
        colA_value = colA[~pd.isnull(colA)].values # null値は除去
        colB_value = colB[~pd.isnull(colB)].values
        row_sim_upper = len(np.intersect1d(colA_value, colB_value))
        row_sim_lower = len(np.union1d(colA_value, colB_value))
        row_sim = float(row_sim_upper) / float(row_sim_lower)
        return row_sim

    @staticmethod
    def col_similarity(tableA, tableB, SM, key_factor):
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


    def existing_method_calc_table_sim_old(self):
        """
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
            sorted_list=sorted(sim_list[nb_name].items(), key=lambda d: d[1], reverse=True)
            if n_q_num > len(sorted_list):
                self.nb_score[nb_name]=0
                continue
            for i in range(min(n_q_num, len(sorted_list))):
                self.nb_score[nb_name]+=sorted_list[i][1]

    def previous_calc_output_sim_old(self, nb_name):
        """
        return True if num of any workflow components is less than query
        """
        ele = "Display_data"
        for display_type in self.query_workflow_info[ele]:
            if display_type not in self.db_workflow_info[nb_name][ele]:
                #logging.info(f"'{display_type}' not in self.db_workflow_info['{nb_name}']['{ele}']")
                return 0
            if self.query_workflow_info[ele][display_type] > self.db_workflow_info[nb_name][ele][display_type]:
                #logging.info(f"self.query_workflow_info['{ele}']['{display_type}'] > self.db_workflow_info['{nb_name}']['{ele}']['{display_type}']")
                return 0
        return 1

    def previous_calc_output_sim_old2(self, nb_name):
        """
        return True if num of any workflow components is less than query
        """
        calc_outputs_start_time = timeit.default_timer()
        score=1
        ele = "Display_data"
        for display_type in self.query_workflow_info[ele]:
            if display_type not in self.db_workflow_info[nb_name][ele]:
                score = 0
                break
            if self.query_workflow_info[ele][display_type] > self.db_workflow_info[nb_name][ele][display_type]:
                score  = 0
                break
        calc_outputs_end_time = timeit.default_timer()
        self.each_calc_time_sum["Display_data"]+=calc_outputs_end_time-calc_outputs_start_time
        return score

    def previous_calc_output_sim(self, nb_name):
        """
        return True if num of any workflow components is less than query
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

    def existing_method_calc_table_sim_particular_nb_old(self, nb_name):
        calc_var_start_time = timeit.default_timer()
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
        if n_q_num > len(sim_list):
            calc_var_end_time = timeit.default_timer()
            self.each_calc_time_sum["Var"]+=calc_var_end_time-calc_var_start_time
            return ret_score

        sorted_list=sorted(sim_list.items(), key=lambda d: d[1], reverse=True)
        visited_n_q=[]
        for i in range(n_q_num):
            if sorted_list[i][0][0] in visited_n_q:
                continue
            visited_n_q.append(sorted_list[i][0][0])
            ret_score+=sorted_list[i][1]
        calc_var_end_time = timeit.default_timer()
        self.each_calc_time_sum["Var"]+=calc_var_end_time-calc_var_start_time
        return ret_score

    def existing_method_calc_table_sim_particular_nb(self, nb_name):
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

    def get_sum_of_table_size(self):
        q_table_size_sum=0
        db_table_size_sum=0
        valid_db_count=0
        invalid_db_count=0
        node_db_list=self.all_node_list["Var"]
        for n_q in self.query_table:
            q_table_size=self.get_table_size3(self.query_table[n_q])
            q_table_size_sum+=q_table_size
            for n_db in node_db_list:
                if self.attr_of_db_node_type[n_db] != self.attr_of_q_node_type[n_q]:
                    continue
                nb_name=self.attr_of_db_nb_name[n_db]
                if nb_name not in self.valid_nb_name:
                    continue

                table_db=self.fetch_var_table(n_db)
                if type(table_db) == type(pd.DataFrame()):
                    db_table_size=self.get_table_size3(table_db)
                    db_table_size_sum+=db_table_size
                    valid_db_count+=1
                else:
                    invalid_db_count+=1

        return q_table_size_sum, db_table_size_sum, valid_db_count, invalid_db_count


    def existing_method_calc_code_sim_old(self):
        """
        セルを頭から結合したソースコードだけをみてtop-kのスコアにする．(self.nb_scoreにセット)
        """
        class_code_relatedness=CodeRelatedness()

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


        for nb_name in self.valid_nb_name:
            # get db nodebook source code
            db_code_table=self.fetch_source_code_table(nb_name)
            db_code_list=db_code_table["cell_code"].values
            db_code="\n".join(db_code_list)
            while "\n\n" in db_code:
                db_code=db_code.replace("\n\n", "\n")
            calc_code_start_time = timeit.default_timer()
            sim=class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
            calc_code_end_time = timeit.default_timer()
            self.each_calc_time_sum["Cell"]+=calc_code_end_time-calc_code_start_time
            self.nb_score[nb_name]=sim
            

    def existing_method_calc_code_sim(self):
        """
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
            sim=class_code_relatedness.calc_code_rel_by_jaccard_index(query_code, db_code)
            calc_code_end_time = timeit.default_timer()
            self.each_calc_time_sum["Cell"]+=calc_code_end_time-calc_code_start_time
            self.nb_score[nb_name]=sim

    def combining_query_code(self):
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
            

    def existing_method_sum(self):
        self.init_each_calc_time_sum()
        self.nb_score={}
        nb_score_according_to_table_sim={}
        if self.w_v!=0:
            self.existing_method_calc_table_sim()
            nb_score_according_to_table_sim=self.nb_score
            self.nb_score={}
        nb_score_according_to_code_sim={}
        if self.w_c!=0:
            self.existing_method_calc_code_sim()
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
            self.nb_score[nb_name]=lib_rel * self.w_l + nb_score_according_to_table_sim[nb_name] * self.w_v + nb_score_according_to_code_sim[nb_name] * self.w_c + output_rel*self.w_d

    def existing_method_sum_fast_method(self, k):
        self.k=k
        self.init_each_calc_time_sum()
        self.nb_score={}
        if not self.flg_running_faster["flg_prune_under_sim"]:
            self.existing_method_sum()
        else:
            if self.w_v==0:
                self.existing_method_sum_fast_method_w_v0()
            else:
                current_nb_score={}
                nb_score_according_to_code_sim={}
                if self.w_c!=0:
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
                    current_nb_score[nb_name] = lib_rel * self.w_l + nb_score_according_to_code_sim[nb_name] * self.w_c
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
                        if self.flg_prune_under_sim_for_new_proposal_method(v_count=n_q_num, current_score=current_nb_score[nb_name], compare_score=k_score, max_v_score=self.w_v):
                            continue
                        self.nb_score[nb_name]=current_nb_score[nb_name]+self.w_v*self.existing_method_calc_table_sim_particular_nb(nb_name)
            
    def existing_method_sum_fast_method_w_v0(self):
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
            
    def get_table_size1(self, table):
        """
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

    #def schema_mapping_with_data_profile(self, colA):
    #    for colB_name in self.data_profile:
    #        colB=self.data_profile[colB_name]
    #        col_sim=self.jaccard_similarity_coefficient(colA, colB)
    #        if col_sim > self.thres_data_profile:
    #            self.data_profile[colB_name]=np.union1d(colA, colB)
    #    pass