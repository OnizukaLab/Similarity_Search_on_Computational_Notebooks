import ast
import json
import logging
import random
import timeit

import networkx as nx
import numpy as np
import pandas as pd

from module2.config import config
from module2.db.schemamapping import SchemaMapping
from module2.db.table_db import generate_graph, pre_vars
from module2.search.search_prov_code import ProvenanceSearch
from module2.search.search_withprov import WithProv
from module2.utils.funclister import FuncLister

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class WithProv_Optimized(WithProv):
    """
    self.(...)でWithProv, SearchTablesのインスタンス変数もとる．

    self.var str:
        self.schema_element (dict{int: dict{str: list[]}):
            テーブルグループごとの，各列名の列の実際のデータ値の重複無し集合．(null値を除く.)
            {テーブルグループID: {列名: [同じ列名を持つグループ内のすべてのテーブルから集めた実際のデータ値]}}
        self.schema_element_sample_row (dict{int: dict{}}):
            self.schema_elementの各テーブルグループに対して指定のサイズ以下の行数となるようにサンプリングしたもの．
        self.schema_element_sample_col (dict{int: dict{str: list[]}):
            self.schema_elementの各テーブルグループに対して指定のサイズ以下の列数，行数となるようにサンプリングしたもの．
        self.Graphs (DataFrame)
        self.n_l2cid (DataFrame)
        self.schema_linking (dict{dict{}})
    """
    ##
    # WithProvとWithProv_Optimizedに複製(全く一致するので関数のオーバーライドでは無い)は,
    # def sketch_query_cols(self, query, sz=10)
    # 
    # 以下はWithProv_Optimizedにある関数.
    # approximate_join_key(tableA, tableB, SM, key, thres_prune): approximate join keyである度合いを計算
    # read_graph_of_notebook(self): SQLデータベースから読み込んでグラフを生成する．
    # __generate_query_node_from_code(self, var_name, code)codeから辞書`dependency`を経由してグラフを作成する．また，変数名からノード名を作成し、グラフにおけるそのノードの隣接ノードとそのラベル集合を返す．
    # __generate_graph(nid, dependency, line2cid): 辞書`dependency`からグラフ(DAG)を生成する
    # __parse_code(code_list): 入力のコードのリストからdependency，line2cid，全てのコード(ただし関数の定義部分は'\n'に変換される)を返す．
    # sample_rows_for_each_column(self, row_size=1000)
    #  __last_line_var(varname, code)
    # sketch_column_and_row_for_meta_mapping(self, sz=5, row_size=1000)
    # sketch_query_cols(self, query, sz=5)
    # index(self)
    # schema_mapping(self, tableA, tableB, meta_mapping, gid)
    # search_additional_training_data(self, query, k, code, var_name, beta, theta)
    # search_alternative_features(self,query,k,code,var_name,alpha,beta,gamma,theta,thres_key_prune,thres_key_cache)
    # search_similar_tables_threshold2(self, query, beta, k, theta, thres_key_cache, thres_key_prune, tflag=False)
    # search_joinable_tables_threshold2(self, query, beta, k, theta, thres_key_cache, thres_key_prune)
    ##
    def __init__(self, dbname, schema=None):
        """ 書き途中
        Args:
            dbname
            schcema (optional)
        """
        super().__init__(dbname, schema) # == SearchTablesの__init__(dbname, schema)を実行

        self.index()

        logging.info("Data Search Extension Prepared!")

    # FIXME: This code is duplicated in search_tables.py
    @staticmethod
    def approximate_join_key(tableA, tableB, SM, key, thres_prune):
        """ 書き途中

        approximate join keyであるかどうかの数値を計算．高いほど当てはまる(おそらく)．

        Args:
            tableA
            tableB
            SM
            key
            thres_prune: scoreAとBの閾値

        Returns:
            float: 閾値thres_pruneより小さければ0を，大きければmax(key_scoreAB, key_scoreBA)を返す．
        """
        # @key_value_A : list
        # @key_value_B : list

        key_value_A = tableA[key].tolist() # .tolist(): リスト型listに変換
        # 多重集合key_value_Aに対しset(key_value_A)は重複無しの集合．
        # (重複無し集合の要素数) / (多重集合の要素数) を計算
        scoreA = float(len(set(key_value_A))) / float(len(key_value_A))
        if scoreA == 1:
            return 1

        kyB = SM[key]
        key_value_B = tableB[kyB].tolist()
        # (重複無し集合の要素数) / (多重集合の要素数) を計算
        scoreB = float(len(set(key_value_B))) / float(len(key_value_B))
        if scoreB == 1:
            return 1

        # scoreAとscoreBが閾値thres_pruneより小さい場合
        if min(scoreA, scoreB) < thres_prune:
            return 0

        # リストkey_value_Aに含まれる要素のうち,リストkey_value_Bにも含まれるものを,配列count_valueAに格納
        count_valueA = []
        for v in key_value_A:
            if v in key_value_B:
                count_valueA.append(v)

        # リストkey_value_Bに含まれる要素のうち,リストkey_value_Aにも含まれるものを,配列count_valueBに格納
        count_valueB = []
        for v in key_value_B:
            if v in key_value_A:
                count_valueB.append(v)

        # (重複無し集合の要素数) / (多重集合の要素数) を計算
        key_scoreAB = float(len(set(count_valueA))) / float(len(count_valueA))
        key_scoreBA = float(len(set(count_valueB))) / float(len(count_valueB))

        return max(key_scoreAB, key_scoreBA)

    def read_graph_of_notebook(self):
        """
        Returns:
            dict: グラフ．
            dict: line2cid_store

        dependency (DataFrame): データベース内のテーブル名"dependen"から読み込んだ内容を格納．
        
        line2cid (DataFrame): データベース内のテーブル名"line2cid"から読み込んだ内容を格納．

        lastliid (DataFrame): データベース内のテーブル名"lastliid"から読み込んだ内容を格納．
        """
        graphs = {}
        # SQLデータベースからテーブル名が"dependen", "line2cid", "lastliid"のテーブルを読み込んでDataFrameに格納する．
        dependency = pd.read_sql_table(
            "dependen", self.eng, schema=config.sql.graph
        )
        line2cid = pd.read_sql_table(
            "line2cid", self.eng, schema=config.sql.graph
        )
        lastliid = pd.read_sql_table("lastliid", self.eng, schema=config.sql.graph)

        dependency_store = {}
        line2cid_store = {}
        lastliid_store = {}

        for index, row in dependency.iterrows():
            dependency_store[row["view_id"]] = json.loads(row["view_cmd"])

        for index, row in line2cid.iterrows():
            line2cid_store[row["view_id"]] = json.loads(row["view_cmd"])

        for index, row in lastliid.iterrows():
            lastliid_store[row["view_id"]] = json.loads(row["view_cmd"])

        for idx, nid in enumerate(dependency_store.keys()): #nid=dependencyの列["view_id"]の1行ごとの値
            try:
                if nid not in lastliid_store:
                    continue

                line_id = lastliid_store[nid]

                if line_id == 0:
                    continue

                nid_name = nid.split("_")[-1]
                Graph = self.__generate_graph(
                    nid_name, dependency_store[nid], line2cid_store[nid]
                )

                if len(list(Graph.nodes)) == 0:
                    continue

                var_name = "_".join(nid.split("_")[1:-1])
                query_name = f"var_{var_name}_{line2cid_store[nid][str(line_id)]}_{nid_name}"
                if Graph.has_node(query_name):
                    query_node = pre_vars(query_name, Graph) #隣接ノード名 list[dict{}]
                    graphs[nid] = query_node
            except Exception as e:
                logging.error(
                    f"Can not generate the graph {idx} due to error {e}"
                )

        return graphs, line2cid_store

    def __generate_query_node_from_code(self, var_name, code):
        """
        codeからdependencyを経由してグラフを作成する．また，変数名からノード名を作成し、グラフにおけるそのノードの隣接ノードとそのラベル集合を返す．

        Args:
            var_name (str): 変数名．例えば"x=4"というコードがあれば変数名は'x'を指す．
            code (str): pythonソースコード

        Return:
            dict{str: str} or dict{str: dict{str: str}}: graphにおけるquery_nameの周辺のノードとラベル集合
        """

        code = "\n".join(
            [t for t in code.split("\\n") if len(t) > 0 and t[0] != "%" and t[0] != "#"]
        )
        code = "'".join(code.split("\\'"))
        code = code.split("\n") #改行で区切って配列にする
        dependency, _, all_code = self.__parse_code(code)
        logging.info(all_code)
        logging.info(dependency)
        line_id = self.__last_line_var(var_name, all_code) # var_nameがcodeの中で一番最後に現れた行番号
        logging.info(line_id)
        graph = generate_graph(dependency) #dict->GiGraph
        logging.info("Output Graph")
        logging.info(list(graph.nodes))

        query_name = "var_" + var_name + "_" + str(line_id) #クエリ名

        query_node = pre_vars(query_name, graph) #graphにおけるquery_nameの周辺のノードとラベル集合
        return query_node

    @staticmethod
    def __generate_graph(nid, dependency, line2cid):
        """
        dependencyをもとに，変数をノードとしたASTのようなグラフを生成する．
        グラフのノードは属性cell_id, line_id, varを持ち，辺はlabelを持つ．
        各変数に対して，それぞれのノードの属性は，cell_idはセル番号，line_idは行番号，varは入力となる変数．
        辺のlabelは，変数に対する操作の内容．
        例えば，{a: {b: add}}ならば，var=b, label=addとなる．

        Args:
            nid
            dependency ({str: {str: str}}): セルのコードの変数間の関係．内容は，{変数1: {変数2: 変数2を入力として変数1に対する操作}}
            line2cid ({int: int}): セルのコードに対応する行番号を, カラの行やdefを考慮して番号1から順に詰めて整理したもの．[変換後の番号: 元の行番号]
        
        Returns:
            DiGraph: NetworkXの有向グラフインスタンス．
        """
        G = nx.DiGraph()
        for i in dependency.keys():
            left = dependency[i][0] # 変数1

            pair_dict = {} # キーは(変数2; 変数2を入力として変数1に対する操作), 値は0.
            right = [] # [[変数2, 変数2を入力として変数1に対する操作]]
            for pa, pb in dependency[i][1]: # pa: 変数2, pb: 変数2を入力として変数1に対する操作
                if f"{pa};{pb}" not in pair_dict:
                    pair_dict[f"{pa};{pb}"] = 0
                    right.append([pa, pb])

            left_node = []
            for ele in left:
                if type(ele) is tuple or type(ele) is list: #変数名が複数ある場合は1つ目をノードの名前に入れる．
                    ele = ele[0]
                left_node.append(f"var_{ele}_{line2cid[i]}_{nid}") # 変数をノードリストに追加 (このリストは下のforループ内で使用)

            for ele in left:
                if type(ele) is tuple or type(ele) is list:
                    ele = ele[0]

                new_node = f"var_{ele}_{line2cid[i]}_{nid}"

                G.add_node(new_node, cell_id=line2cid[i], line_id=i, var=ele) #グラフGにノードを追加

                for dep, ename in right: # dep: 変数2, ename: 変数2を入力として変数1に対する操作
                    candidate_list = G.nodes # グラフのすべてのノード
                    rankbyline = [] # [[str, int]]: 変数のノード名とその変数の行番号のタプルのリスト.
                    for cand in candidate_list:
                        if G.nodes[cand]["var"] == dep: #グラフのあるノードに対して変数depは変数2(操作の入力の変数)であったとき
                            if cand in left_node: # 今回保存しようとしているdependencyの変数1のうちの1つであった場合
                                continue
                            rankbyline.append((cand, int(G.nodes[cand]["line_id"])))
                    rankbyline = sorted(rankbyline, key=lambda d: d[1], reverse=True) # 行番号で降順にソート

                    if len(rankbyline) == 0: # rightが1つも無い場合(例えば, a=1+2の行など)
                        if dep not in ["np", "pd"]:
                            candidate_node = (
                                "var_" + dep + "_" + str(1) + "_" + str(nid)
                            )
                            G.add_node(candidate_node, cell_id=0, line_id=1, var=dep)
                        else:
                            candidate_node = dep + str(nid)
                            G.add_node(candidate_node, cell_id=0, line_id=1, var=dep)

                    else:
                        candidate_node = rankbyline[0][0] # 変数が最後に出てきた行番号を含むノード名

                    G.add_edge(new_node, candidate_node, label=ename) #辺を追加

        return G

    # FIXME: Duplicated code in store_prov.py
    @staticmethod
    def __parse_code(code_list):
        """
        コードを整形し抽象度の高いASTにする．

        Args:
            code_list ([str]): セルのコードを1行ごとに切って各添字に格納した配列．

        Returns:
            dict {[str], [str, str]}: FuncLister.dependency
            dict [int: int]:
                内容は[変換後の番号: 元の行番号]．
                セルのコードに対応する行番号を, カラの行やdefを考慮して番号1から順に詰めて整理したもの．
                'def'と'return'が入っている行はスキップされる
            str : コード全体の文字列．ただし関数の定義部分は'\n'に変換されている．
        """
        test = FuncLister()
        all_code = ""
        line2cid = {} #ライン番号をコード番号に変換

        lid = 1 #line ID
        fflg = False # defとreturnの行の間だけTrue
        for cid, cell in enumerate(code_list):
            codes = cell.split("\\n")
            new_codes = []
            for code in codes:
                if code[:3].lower() == "def": #defの行
                    fflg = True
                    continue

                temp_code = code.strip(" ")
                temp_code = temp_code.strip("\t")

                if temp_code[:6].lower() == "return": #returnの行
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
                    logging.info(code)

            all_code = all_code + "\n".join(new_codes) + "\n"

        all_code = all_code.strip("\n")

        tree = ast.parse(all_code)
        test.visit(tree)
        return test.dependency, line2cid, all_code

    def sample_rows_for_each_column(self, row_size=1000):
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

    @staticmethod
    def __last_line_var(varname, code):
        """
        codeの中で最後に出てくるvarnameの位置(行番号)を返す．

        Args:
            varname (str): 最後に出てくる位置を調べたい変数名．
            code (str): コード全体の文字列．

        Return:
            int: codeの中で最後に出てくるvarnameの行番号．
        """
        ret = 0
        code = code.split("\n")
        for id, i in enumerate(code):
            if "=" not in i:
                continue
            j = i.split("=")
            j = [t.strip(" ") for t in j]

            if varname in j[0]:
                if varname == j[0][-len(varname) :]:
                    ret = id + 1 #idは0から始まるがretは1から始まるので+1する
        return ret

    def sketch_column_and_row_for_meta_mapping(self, sz=5, row_size=1000):
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
                        self.schema_element_sample_col[i][sc] = self.schema_element[i][
                            sc
                        ]
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

    def sketch_query_cols(self, query, sz=5):
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

    def index(self):
        self.sample_rows_for_each_column() # schema_elementの各テーブルから, 各列の行数がrow_size以下になるよう行をサンプリングしてschema_element_sample_rowを構成
        self.sketch_column_and_row_for_meta_mapping()
        logging.info("Reading Graph of Notebooks.")
        self.Graphs, self.n_l2cid = self.read_graph_of_notebook()

    def schema_mapping(self, tableA, tableB, meta_mapping, gid):
        """
        tableAとtableBのスキーママッピングの値を計算する．
        スキーママッピングの値とは，2テーブルの共通列名の列それぞれの
        データの類似度(row_sim)のうち，最大の類似度のことである．

        Args:
            tableA (DataFrame): スキーママッピングを調べたいテーブル．
            tableB (DataFrame): スキーママッピングを調べたいテーブル．
            meta_mapping (dict{int: dict{str: list[]}): 
                内容は，{テーブルグループID: {列名: [同じ列名を持つグループ内のすべてのテーブルから集めた実際のデータ値]}}.
            gid (int):
                テーブルグループID．従属関係にあるテーブルは同じグループIDをもつ．
                (変数A_xとB_yが従属関係にあるとは，変数A_xが含まれているセルから，ワークフローグラフ上で到達可能なセルが変数B_yを持つことであるとする．
                ここでxとyは，ノートブック内で複数回変数が登場するときのバージョンの違い(実際は行番号の値?)を表す．)
        
        Returns:
            dict{int: str}: tableAとtableBで共通する列のSID(Schema ID?)と列名のリスト(辞書).内容は{sid: 列名}
            float: tableAとtableBの列の類似度(row_similarity)のうち，最大の値のもの．共通の列が存在しない場合は0．

        self.var str:
            self.schema_linking (dict{int: dict{str: int}}):
                テーブルグループそれぞれに対し，列名とsid(列名ごとの固有ID.schema IDの略?)の対応関係が格納されている．
                {テーブルグループID: {列名: sid}}
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

    def search_additional_training_data(self, query, k, code, var_name, beta, theta):
        """
        Additional Training/Validation Tables検索を実行すると実質最初に呼び出される．

        Args:
            query (DataFrame): クエリ(テーブル).
            k (int)
            code
            var_name (str)
            beta (float or num-like)
            theta (float)
        """

        # introduce the schema mapping class
        self.index()

        SM_test = SchemaMapping() # default sim_thres=0.3

        # choose only top possible key columns
        query_col_valid = self.sketch_query_cols(query) # query_col_valid (list[str]): queryから5列をサンプリングした列名リスト.

        # do partial schema mapping
        partial_mapping = SM_test.mapping_naive_tables(
            query,
            query_col_valid,
            self.schema_element_sample_col,
            self.schema_element_dtype,
        )

        unmatched = {}
        for i in partial_mapping.keys():
            unmatched[i] = {}
            for j in query.columns.tolist(): # .tolist(): リスト型listに変換
                unmatched[i][j] = {}
                if (j in query_col_valid) and (j not in partial_mapping[i]):
                    for l in self.schema_element[i].keys():
                        unmatched[i][j][l] = ""

        prov_class = ProvenanceSearch(self.Graphs)
        query_node = self.__generate_query_node_from_code(var_name, code)
        table_prov_rank = prov_class.search_score_rank(query_node)
        table_prov_score = {}

        for i, j in table_prov_rank:
            table_prov_score["rtable" + i] = j
        logging.info(table_prov_score)

        top_tables = []
        rank_candidate = []
        rank2 = []

        for i in self.real_tables.keys():
            tname = i
            if tname not in table_prov_score:
                logging.info(tname)
                continue
            else:
                gid = self.table_group[tname[6:]]
                if gid not in partial_mapping:
                    continue

                tableS = query
                tableR = self.real_tables[i]
                SM, ms = self.schema_mapping(tableS, tableR, partial_mapping, gid)
                rank_candidate.append(
                    (tname, float(1) / float(table_prov_score[tname] + 1), SM)
                )

                upp_col_sim = float(min(tableS.shape[1], tableR.shape[1])) / float(
                    max(tableS.shape[1], tableR.shape[1])
                )
                rank2.append(upp_col_sim)

        rank_candidate = sorted(rank_candidate, key=lambda d: d[1], reverse=True)
        rank2 = sorted(rank2, reverse=True)

        if len(rank_candidate) == 0:
            return []

        if len(rank_candidate) > k:
            ks = k
        else:
            ks = len(rank_candidate)

        for i in range(ks):
            tableS = query
            tableR = self.real_tables[rank_candidate[i][0]]
            gid = self.table_group[rank_candidate[i][0][6:]]
            SM_real = rank_candidate[i][2]
            (
                SM_real,
                meta_mapping,
                unmatched,
                sm_time,
            ) = SM_test.mapping_naive_incremental(
                query,
                tableR,
                gid,
                partial_mapping,
                self.schema_linking,
                unmatched,
                mapped=SM_real,
            )
            score = (
                float(beta) * self.col_similarity(tableS, tableR, SM_real, 1)
                + float(1 - beta) * rank_candidate[i][1]
            )
            top_tables.append((rank_candidate[i][0], score))

        top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)

        min_value = top_tables[-1][1]

        ks = ks - 1
        id = 0
        while True:
            if ks + id >= len(rank_candidate):
                break

            threshold = (
                float(beta) * rank2[ks + id]
                + float(1 - beta) * rank_candidate[ks + id][1]
            )

            if threshold <= min_value * theta:
                break
            else:
                id = id + 1
                if ks + id >= len(rank_candidate):
                    break

                tableR = self.real_tables[rank_candidate[ks + id][0]]
                gid = self.table_group[rank_candidate[ks + id][0][6:]]
                SM_real = rank_candidate[ks + id][2]
                (
                    SM_real,
                    meta_mapping,
                    unmatched,
                    sm_time,
                ) = SM_test.mapping_naive_incremental(
                    query,
                    tableR,
                    gid,
                    partial_mapping,
                    self.schema_linking,
                    unmatched,
                    mapped=SM_real,
                )

                new_score = (
                    float(beta) * self.col_similarity(query, tableR, SM_real, 1)
                    + float(1 - beta) * rank_candidate[i][1]
                )

                if new_score <= min_value:
                    continue
                else:
                    top_tables.append((rank_candidate[ks + id][0], new_score))
                    top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
                    min_value = top_tables[ks][1]

        # logging.info("Schema Mapping Costs: %s Seconds" % time1)
        # logging.info("Full Search Costs: %s Seconds" % time3)

        rtables_names = self.remove_dup2(top_tables, ks)

        rtables = []
        for i in rtables_names:
            rtables.append((i, self.real_tables[i]))

        return rtables

    def search_alternative_features(
        self,
        query,
        k,
        code,
        var_name,
        alpha,
        beta,
        gamma,
        theta,
        thres_key_prune,
        thres_key_cache,
    ):

        """
        Alternative Feature Tables検索を実行すると実質最初に呼び出される．
        引数のサンプル：query_table, 10, code, var_name, 90, 200, 0.1, 10, 0.9, 0.2
        """

        # choose only top possible key columns
        query_col = self.sketch_query_cols(query)

        # introduce the schema mapping class
        SM_test = SchemaMapping()
        # do partial schema mapping
        partial_mapping = SM_test.mapping_naive_tables(
            query, query_col, self.schema_element_sample_col, self.schema_element_dtype
        )

        unmatched = {}
        for i in partial_mapping.keys():
            unmatched[i] = {}
            for j in query.columns.tolist(): # .tolist(): リスト型listに変換
                unmatched[i][j] = {}
                if (j in query_col) and (j not in partial_mapping[i]):
                    for l in self.schema_element[i].keys():
                        unmatched[i][j][l] = ""

        self.query_fd = {}
        self.already_map = {}
        for i in self.schema_linking.keys():
            self.already_map[i] = {}

        prov_class = ProvenanceSearch(self.Graphs)
        query_node = self.__generate_query_node_from_code(var_name, code)

        table_prov_rank = prov_class.search_score_rank(query_node)
        table_prov_score = {}
        for i, j in table_prov_rank:
            table_prov_score["rtable" + i.lower()] = j

        logging.info(table_prov_score)

        top_tables = []
        rank_candidate = []
        rank2 = []

        tableS = query
        for i in self.real_tables.keys():
            tname = i

            if tname not in table_prov_score:
                continue
            else:
                logging.info(tname)
                gid = self.table_group[tname[6:]]
                if gid not in partial_mapping:
                    continue

                tableR = self.real_tables[i]

                SM, ms = self.schema_mapping(tableS, tableR, partial_mapping, gid)

                if len(SM.items()) == 0:
                    continue

                tableSnotintableR = []
                for sk in tableS.columns.tolist(): # .tolist(): リスト型listに変換
                    if sk not in SM:
                        tableSnotintableR.append(sk)

                upper_bound_col_score1 = float(1) / float(
                    len(tableR.columns.values) + len(tableSnotintableR)
                )

                upper_bound_col_score = upper_bound_col_score1 + float(
                    min(len(tableS.columns.tolist()), len(tableR.columns.tolist())) - 1
                ) / float(len(tableR.columns.values) + len(tableSnotintableR) - 1)

                upper_bound_row_score = ms / float(
                    abs(tableR.shape[0] - tableS.shape[0]) + 1
                )

                rank2.append(
                    float(alpha) * upper_bound_col_score
                    + float(beta) * upper_bound_row_score
                )

                rank_candidate.append((tname, float(table_prov_score[tname]), SM))

        rank2 = sorted(rank2, reverse=True)
        rank_candidate = sorted(rank_candidate, key=lambda d: d[1], reverse=True)

        if len(rank_candidate) == 0:
            return []

        if len(rank_candidate) > k:
            ks = k
        else:
            ks = len(rank_candidate)

        for i in range(ks):
            tableR = self.real_tables[rank_candidate[i][0]]
            gid = self.table_group[rank_candidate[i][0][6:]]
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
                partial_mapping,
                self.schema_linking,
                thres_key_prune,
                thres_key_cache,
                unmatched,
            )

            score = (
                float(alpha) * (col_sim)
                + float(beta)
                * row_sim
                / float(abs(tableR.shape[0] - tableS.shape[0]) + 1)
                + float(gamma) * rank_candidate[i][1]
            )

            logging.info(rank_candidate[i][0])
            logging.info(col_sim * alpha)
            logging.info(
                row_sim * beta / float(abs(tableR.shape[0] - tableS.shape[0]) + 1)
            )
            logging.info(rank_candidate[i][1] * gamma)
            logging.info(score)

            logging.info("\n")
            top_tables.append((rank_candidate[i][0], score, key_chosen))

        top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
        min_value = top_tables[-1][1]

        ks = ks - 1
        id = 0
        while True:

            if ks + id >= len(rank_candidate):
                break

            threshold = float(gamma) * rank_candidate[ks + id][1] + rank2[ks + id]

            if threshold <= min_value * theta:
                break
            else:
                id = id + 1
                if ks + id >= len(rank_candidate):
                    break

                tableR = self.real_tables[rank_candidate[ks + id][0]]
                gid = self.table_group[rank_candidate[ks + id][0][6:]]
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
                    partial_mapping,
                    self.schema_linking,
                    thres_key_prune,
                    thres_key_cache,
                    unmatched,
                )
                new_score = (
                    float(alpha) * (col_sim)
                    + float(beta)
                    * row_sim
                    / float(abs(tableR.shape[0] - tableS.shape[0]) + 1)
                    + float(gamma) * rank_candidate[ks + id][1]
                )
                logging.info(rank_candidate[ks + id][0])
                logging.info(col_sim * alpha)
                logging.info(
                    row_sim * beta / float(abs(tableR.shape[0] - tableS.shape[0]) + 1)
                )
                logging.info(rank_candidate[ks + id][1] * gamma)
                logging.info(new_score)

                logging.info("\n")

                if new_score <= min_value:
                    continue
                else:
                    top_tables.append(
                        (rank_candidate[ks + id][0], new_score, key_chosen)
                    )
                    top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
                    min_value = top_tables[ks][1]

        rtables_names = self.remove_dup(top_tables, ks)
        rtables = []
        for i, j in rtables_names:
            rtables.append((i, self.real_tables[i]))

        return rtables

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

        time1 = 0
        start_time = timeit.default_timer()
        # Do mapping
        meta_mapping = SM_test.mapping_naive_tables(
            self.query,
            query_col,
            self.schema_element_sample_col,
            self.schema_element_dtype,
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
                    for l in self.schema_element_sample_row[i].keys():
                        unmatched[i][j][l] = ""

        top_tables = []
        Cache_MaxSim = {}

        rank2 = []
        rank_candidate = []

        for i in self.real_tables.keys():

            tname = i
            gid = self.table_group[tname[6:]]
            if gid not in meta_mapping:
                continue

            tableS = self.query
            tableR = self.real_tables[i]

            start_time = timeit.default_timer()
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
            tableR = self.real_tables[rank_candidate[i][0]]
            gid = self.table_group[rank_candidate[i][0][6:]]
            SM_real = rank_candidate[i][2]
            (
                score,
                meta_mapping,
                unmatched,
                sm_time,
                key_chosen,
            ) = self.comp_table_similarity_key(
                SM_test,
                self.query,
                tableR,
                beta,
                SM_real,
                gid,
                meta_mapping,
                self.schema_linking,
                thres_key_prune,
                thres_key_cache,
            )
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

                tableR = self.real_tables[rank_candidate[ks + id][0]]
                gid = self.table_group[rank_candidate[ks + id][0][6:]]
                SM_real = rank_candidate[ks + id][2]
                (
                    rs,
                    meta_mapping,
                    unmatched,
                    sm_time,
                    key_chosen,
                ) = self.comp_table_similarity_key(
                    SM_test,
                    self.query,
                    tableR,
                    beta,
                    SM_real,
                    gid,
                    meta_mapping,
                    self.schema_linking,
                    thres_key_prune,
                    thres_key_cache,
                    unmatched,
                )
                time1 += sm_time
                new_score = rs

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

    def search_joinable_tables_threshold2(
        self, query, beta, k, theta, thres_key_cache, thres_key_prune
    ):
        """
        Joinable Tables検索を実行すると実質最初に呼び出される．


        Args:
            query (DataFrame): テーブル.
            beta (float): parameter β allows us to adjust the weight on row and column terms(Ref: 論文)
            k (int): 類似度上位k件の指定.
            theta
            thres_key_cache: ?
            thres_key_prune

        Returns:
            list[str, DataFrame]: スコアが高い順にk個の[テーブル名, 実際のテーブル]
        
        self.var str:
            self.query (DataFrame): テーブル.
            self.query_fd: ?
            self.already_map (dict{int: dict{}}): {グループID: {}}
            self.schema_linking (dict{int: dict{str: int}}):
                テーブルグループそれぞれに対し，列名とsid(列名ごとの固有ID.schema IDの略?)の対応関係が格納されている．
                {テーブルグループID: {列名: sid}}
            self.real_tables ({str: DataFrame}): (テーブル名: 実際のテーブルの内容)の辞書．
        """

        self.query = query
        self.query_fd = {}
        self.already_map = {}

        for i in self.schema_linking.keys(): # 各グループIDに対して
            self.already_map[i] = {}

        query_col = self.sketch_query_cols(query) # list[str]

        SM_test = SchemaMapping()

        # 変数unmatchedの初期化
        unmatched = {}
        for i in self.schema_linking.keys(): # 各グループIDに対して
            unmatched[i] = {}
            for j in query.columns.tolist(): # .tolist(): リスト型listに変換
                unmatched[i][j] = {} 

        start_time1 = timeit.default_timer() # 時間計測用

        time1 = 0 # 時間計測用
        start_time = timeit.default_timer() # 時間計測用
        meta_mapping, unmatched = SM_test.mapping_naive_tables_join( # meta_mapping: {グループID: {列名A: 列名B}}
            self.query,
            query_col,
            self.schema_element_sample_col,
            self.schema_element_sample_row,
            self.schema_element_dtype,
            unmatched,
        )
        end_time = timeit.default_timer() # 時間計測用
        time1 += end_time - start_time # 時間計測用
        # logging.info(str(meta_mapping))
        logging.info(
            "Initial Schema Mapping Costs: %s Seconds." % (end_time - start_time)
        )

        top_tables = []
        Cache_MaxSim = {} # {str: float}: {テーブル名: self.queryとの列の類似度の最大値}

        rank2 = [] # list[float]: sim^{mu}_{beta}(S,T)の値のリスト.
        rank_candidate = [] #list[str, float, dict{int: str}]: [テーブル名, vname_score, dict{SID(列のID): 列名}]

        # table overlapを計算. 結果はrank2, rank_candidateに格納.
        for i in self.real_tables.keys(): # i (str): テーブル名

            tname = i # tname (str): テーブル名
            gid = self.table_group[tname[6:]] # gid (int):グループID
            if gid not in meta_mapping:
                continue

            tableS = self.query # tableS (DataFrame)
            tableR = self.real_tables[i] # tableR (DataFrame)

            start_time = timeit.default_timer()
            # SM dict{int: str}: tableSとtableBR共通する列のSID(Schema ID?)と列名のリスト(辞書).内容は{sid: 列名}
            # ms (float): sim^{mu}_{row}(tableS,tableR)
            SM, ms = self.schema_mapping(tableS, tableR, meta_mapping, gid)
            end_time = timeit.default_timer()
            time1 = time1 + end_time - start_time

            Cache_MaxSim[tname] = ms # ms = sim^{mu}_{row}(S,T)

            if len(SM.items()) == 0:
                continue

            tableSnotintableR = [] # list[str]: スキーママッピングSMに含まれないqueryの列名のリスト
            for sk in tableS.columns.tolist(): # .tolist(): リスト型listに変換
                if sk not in SM:
                    tableSnotintableR.append(sk)

            vname_score = float(1) / float(
                len(tableR.columns.values) + len(tableSnotintableR) #tableSとtableRの重複無し列名の数
            )
            # 論文の式(1)または式(11)? vname_score2 = sim^{mu}_{row}(S,T)
            vname_score2 = float(
                max(len(tableR.columns.values), len(tableS.columns.values)) - 1
            ) / float(len(tableR.columns.values) + len(tableSnotintableR))
            # 論文の式(17) ubound = sim^{mu}_{beta}(S,T)
            ubound = beta * vname_score2 + float(1 - beta) * Cache_MaxSim[tname]

            rank2.append(ubound)
            rank_candidate.append((tname, vname_score, SM))

        # 類似度でソート
        rank2 = sorted(rank2, reverse=True)
        rank_candidate = sorted(rank_candidate, key=lambda d: d[1], reverse=True)

        if len(rank_candidate) == 0:
            return []

        if len(rank_candidate) > k:
            ks = k
        else:
            ks = len(rank_candidate)

        for i in range(ks):
            tableR = self.real_tables[rank_candidate[i][0]] # tableR (DataFrame)
            SM_real = rank_candidate[i][2]
            gid = self.table_group[rank_candidate[i][0][6:]]
            (
                score,
                meta_mapping,
                unmatched,
                sm_time,
                key_chosen,
            ) = self.comp_table_joinable_key(
                SM_test,
                self.query,
                tableR,
                beta,
                SM_real,
                gid,
                meta_mapping,
                self.schema_linking,
                thres_key_prune,
                unmatched,
            )
            top_tables.append((rank_candidate[i][0], score, key_chosen)) # triple(テーブル名, score, 列リスト)
            time1 += sm_time

        top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True)
        min_value = top_tables[-1][1]

        ks = ks - 1
        id = 0
        while True:
            if ks + id >= len(rank_candidate):
                break

            threshold = beta * rank_candidate[ks + id][1] + rank2[0]

            if threshold <= min_value * theta:
                break
            else:
                id = id + 1
                if ks + id >= len(rank_candidate):
                    break

                tableR = self.real_tables[rank_candidate[ks + id][0]]
                SM_real = rank_candidate[ks + id][2]
                gid = self.table_group[rank_candidate[ks + id][0][6:]]
                (
                    new_score,
                    meta_mapping,
                    unmatched,
                    sm_time,
                    key_chosen,
                ) = self.comp_table_joinable_key(
                    SM_test,
                    self.query,
                    tableR,
                    beta,
                    SM_real,
                    gid,
                    meta_mapping,
                    self.schema_linking,
                    thres_key_prune,
                    unmatched,
                )
                time1 += sm_time

                if new_score <= min_value:
                    continue
                else:
                    top_tables.append( # #list[str, float, int]: [テーブル名, score, SID]
                        (rank_candidate[ks + id][0], new_score, key_chosen)
                    )
                    top_tables = sorted(top_tables, key=lambda d: d[1], reverse=True) #scoreで降順にソート
                    min_value = top_tables[ks][1]

        end_time1 = timeit.default_timer()
        time3 = end_time1 - start_time1

        rtables_names = self.remove_dup(top_tables, k)

        logging.info("Schema Mapping Costs: %s Seconds" % time1)
        logging.info("Full Search Costs:%s Seconds" % time3)

        rtables = []
        for i, j in rtables_names: # i (str): テーブル名
            # print(i,j)
            rtables.append((i, self.real_tables[i])) # [str, DataFrame]: [テーブル名, 実際のテーブル]

        return rtables
