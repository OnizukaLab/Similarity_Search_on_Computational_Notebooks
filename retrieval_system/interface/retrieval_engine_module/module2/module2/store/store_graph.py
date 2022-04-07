import base64
import logging
import json

import pandas as pd
from py2neo import Node, Relationship, NodeMatcher,RelationshipMatcher
from module2.config import config


class ProvenanceStorage:
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
    def __init__(self, postgres_eng, graph_eng):
        """
        PostgreSQLのインスタンスとpy2neoのインスタンスをインスタンスにセットする．

        Args:
            postgres_eng: PostgreSQLのインスタンス
            graph_eng (Graph): py2neoのクラス`Graph`のインスタンス．インスタンス変数graph_dbに格納．
        """
        self.graph_db = graph_eng
        self.postgres_eng = postgres_eng
        self.code_dict = {}

    def _initialize_code_dict(self):
        query1 = "DROP SCHEMA IF EXISTS " + config.sql.provenance + " CASCADE;" # schema名: "nb_provenance"
        query2 = "CREATE SCHEMA " + config.sql.provenance + ";"
        query3 = "CREATE TABLE " + config.sql.provenance + ".code_dict (code VARCHAR(1000), cell_id INTEGER);"

        with self.postgres_eng.connect() as conn:
            try:
                conn.execute(query1)
            except Exception as e:
                logging.error(
                    f"Store Provenance: dropping of provenance schema failed with error {e}"
                )

            try:
                conn.execute(query2)
                conn.execute(query3)
            except Exception as e:
                logging.error(
                    f"Store Provenance: creation of provenance schema failed with error {e}"
                )

    def _initialize_lib_db(self):
        query1 = "CREATE SCHEMA IF NOT EXISTS " + config.sql.nb_info + ";"
        query2 = "CREATE TABLE IF NOT EXISTS " + config.sql.nb_info + ".nb_libraries (nb_name VARCHAR(1000), libraries VARCHAR(100));"

        with self.postgres_eng.connect() as conn:
            try:
                conn.execute(query1)
            except Exception as e:
                logging.error(
                    f"Store Libraries imported in NB: dropping of libraries schema failed with error {e}"
                )
            try:
                conn.execute(query2)
            except Exception as e:
                logging.error(
                    f"Store Libraries imported in NB: dropping of libraries table failed with error {e}"
                )

    
    def _initialize_display_type_db(self):
        query1 = "CREATE SCHEMA IF NOT EXISTS " + config.sql.nb_info + ";"
        query2 = "CREATE TABLE IF NOT EXISTS " + config.sql.nb_info + ".display_type (nb_name VARCHAR(1000), cell_id INTEGER, data_type VARCHAR(100));"

        with self.postgres_eng.connect() as conn:
            try:
                conn.execute(query1)
            except Exception as e:
                logging.error(
                    f"Store Display data type imported in NB: dropping of display_type schema failed with error {e}"
                )
            try:
                conn.execute(query2)
            except Exception as e:
                logging.error(
                    f"Store Display data type imported in NB: dropping of display_type table failed with error {e}"
                )

    def _initialize_nottablevar_db(self):
        query1 = "CREATE SCHEMA IF NOT EXISTS " + config.sql.nottablevar + ";"
        with self.postgres_eng.connect() as conn:
            try:
                conn.execute(query1)
            except Exception as e:
                logging.error(
                    f"Store NotTableVar: dropping of schema failed with error {e}"
                )
        
    #不使用
    def _initialize_code_dict2(self):
        query1 = "DROP SCHEMA IF EXISTS " + config.sql.provenance + " CASCADE;" # schema名: "nb_provenance"
        query2 = "CREATE SCHEMA " + config.sql.provenance + ";"
        query3 = "CREATE TABLE " + config.sql.provenance + ".code_dict2 (code VARCHAR(1000), cell_id INTEGER, nb_name VARCHAR(1000), real_cell_id INTEGER);"

        with self.postgres_eng.connect() as conn:
            try:
                conn.execute(query1)
            except Exception as e:
                logging.error(
                    f"Store Provenance: dropping of provenance schema failed with error {e}"
                )

            try:
                conn.execute(query2)
                conn.execute(query3)
            except Exception as e:
                logging.error(
                    f"Store Provenance: creation of provenance schema failed with error {e}"
                )

    def _fetch_code_dict(self):
        """
        データベースからPostgreSQLでデータを取ってきて，インスタンス変数code_dict (ProvenanceStorage.code_dict) に格納する．
        """
        with self.postgres_eng.connect() as conn:
            try:
                code_table = pd.read_sql_table(
                    "code_dict", conn, schema=config.sql.provenance
                )
                for index, row in code_table.iterrows():
                    self.code_dict[row["code"]] = int(row["cell_id"])
            except Exception as e:
                logging.error(
                    f"Store Provenance: reading code table failed with error {e}"
                )

    def _store_code_dict(self):
        """
        データベースにPostgreSQLでセル(ソースコードと対応するセルID)のデータを格納する．
        格納するデータはインスタンス変数code_dict (ProvenanceStorage.code_dict) の中身．
        データベースのスキーマ: (code, cell_id)
        """
        dict_store = {"code": [], "cell_id": []}
        for i in self.code_dict.keys():
            dict_store["code"].append(i)
            dict_store["cell_id"].append(self.code_dict[i])
        dict_store_code = pd.DataFrame.from_dict(dict_store) #DataFrameの構造にする
        with self.postgres_eng.connect() as conn:
            dict_store_code.to_sql( # DataFrameの構造をpandasの操作`.to_sql`を使ってデータベースに渡す．
                "code_dict", conn, schema=config.sql.provenance, if_exists="replace", index=False
            )
    #不使用
    def _store_code_dict2(self, nb_name, real_cell_id):
        """
        _initialize_code_dict2(self)を利用して初期化したときのみ有効
        テーブル名code_dict2
        """
        dict_store = {"code": [], "cell_id": []}
        for i in self.code_dict.keys():
            dict_store["code"].append(i)
            dict_store["cell_id"].append(self.code_dict[i])
            dict_store["nb_name"].append(nb_name)
            dict_store["real_cell_id"].append(real_cell_id)
        dict_store_code = pd.DataFrame.from_dict(dict_store) #DataFrameの構造にする
        with self.postgres_eng.connect() as conn:
            dict_store_code.to_sql( # DataFrameの構造をpandasの操作`.to_sql`を使ってデータベースに渡す．
                "code_dict2", conn, schema=config.sql.provenance, if_exists="replace", index=False
            )

    def add_cell(self, code, prev_node, var, cell_id, nb_name):
        """
        Stores the Jupyter cell in base64 encoded form, and adds a link to the previously
        stored node (for cell provenance tracking).
        ワークフロー(?)グラフにセルのノードを追加する．現在のセルに対応するNodeインスタンスを返す．
        [グラフ] グラフデータベースの方にも同様にして，セルのコードとセルIDのペアを保存．
        [グラフ] さらに現在のセルと引数prev_nodeで指定されるノードが互いに`successor`，`parent`のリレーションを設定し，Neo4jでデータベースに入れる．
        [グラフ] セルに対して，変数がそのセルに含まれていることをグラフのリレーションで保存．(リレーション`Contain`および`Containedby`)
        [PostgreSQL] インスタンス変数code_dictを経由して，セルのコードとセルIDのペアをPostgreSQLでデータベースにの内容を保存．

        Args:
            code (str): セルの実際のコード．
            prev_node (Node): グラフ上で新しく作るセルのノードの親としたいセルのノード．
            var (str): 1つのテーブルを保持する変数名．
            cell_id (int)
            nb_name (str)

        Returns:
            Node: 引数`code`をノードとしてグラフに追加したときの，ノード．
        """
        self._fetch_code_dict()
        bcode = base64.b64encode(bytes(code, "utf-8")) #UTF-8からBase64にエンコード
        nbcode = bcode.decode("utf-8") #Base64からUTF-8にデコード (文字の種類を制限するためにこのような操作？)
        matcher = NodeMatcher(self.graph_db) #matcherの初期化

        if bcode in self.code_dict or nbcode in self.code_dict:
            # すでにgraph_dbにcodeが保存されていたとき
            current_cell = matcher.match("Cell", source_code=nbcode).first() # セルのソースコードがBase64の形式で一致するノードを検索
            #current_cell = matcher.match("Cell", source_code=bcode).first() # セルのソースコードがBase64の形式で一致するノードを検索
        else:
            if len(list(self.code_dict.values())) != 0: # code_dictに1つ以上コードが格納されているとき．
                max_id = max(list(self.code_dict.values())) #max_idはcode_dict.values()のうち(おそらくセル番号？)，最も大きい値を返す．
            else:
                max_id = 0
            #current_cell = Node("Cell", name=f"cell_{max_id + 1}", source_code=bcode) #セル番号をインクリメントして，Base64にエンコードされたコードをkeyがsource_codeのvalueに入れる
            current_cell = Node("Cell", name=f"cell_{max_id + 1}", source_code=nbcode) #Base64にエンコードしたコードだとうまくいかない．
            self.graph_db.create(current_cell) #エラー
            self.graph_db.push(current_cell)

            if prev_node is not None:
                # prev_node(ノード)はcurrent_cell(ノード)のSuccessor(辺のラベル)というリレーションを設定, データベースに入れる.
                successor = Relationship(prev_node, "Successor", current_cell)
                self.graph_db.create(successor)
                self.graph_db.push(successor)
                # リレーションをデータベースに入れる. リレーション: current_cellはprev_nodeのParent
                parent = Relationship(current_cell, "Parent", prev_node)
                self.graph_db.create(parent)
                self.graph_db.push(parent)

            self.code_dict[bcode] = max_id + 1 # セル番号のインクリメント
            #self.code_dict[nbcode] = max_id + 1 # セル番号のインクリメント
            try:
                self._store_code_dict() # PostgreSQLでデータベースにインスタンス変数code_dictの内容を保存
            except Exception as e:
                logging.info(f"Code update for Neo4j failed with error {e}")

        var_name = f"{cell_id}_{var}_{nb_name}" # グラフにおける変数のノード名を設定
        current_var = matcher.match("Var", name=var_name).first()

        # 各セルに含まれる変数をグラフ形式で保存．
        if current_var is None: #current_varのノードがまだ生成(保存)されていない場合
            current_var = Node("Var", name=var_name)

            self.graph_db.create(current_var)
            self.graph_db.push(current_var)

            # リレーションを設定． 各セルに対してこの変数が含まれることをリレーションとして設定する．
            contains_edge = Relationship(current_cell, "Contains", current_var)
            self.graph_db.create(contains_edge)
            self.graph_db.push(contains_edge)
            contained_by_edge = Relationship(current_var, "Containedby", current_cell)
            self.graph_db.create(contained_by_edge)
            self.graph_db.push(contained_by_edge)

        return current_cell


    def add_cell_no_table(self, code, prev_node, var, cell_id, nb_name):
        """
        Stores the Jupyter cell in base64 encoded form, and adds a link to the previously
        stored node (for cell provenance tracking).
        ワークフロー(?)グラフにセルのノードを追加する．現在のセルに対応するNodeインスタンスを返す．
        [グラフ] グラフデータベースの方にも同様にして，セルのコードとセルIDのペアを保存．
        [グラフ] さらに現在のセルと引数prev_nodeで指定されるノードが互いに`successor`，`parent`のリレーションを設定し，Neo4jでデータベースに入れる．
        [グラフ] セルに対して，変数がそのセルに含まれていることをグラフのリレーションで保存．(リレーション`Contain`および`Containedby`)
        [PostgreSQL] インスタンス変数code_dictを経由して，セルのコードとセルIDのペアをPostgreSQLでデータベースにの内容を保存．

        Args:
            code (str): セルの実際のコード．
            prev_node (Node): グラフ上で新しく作るセルのノードの親としたいセルのノード．
            var (str)
            cell_id (int)
            nb_name (str)

        Returns:
            Node: 引数`code`をノードとしてグラフに追加したときの，ノード．
        """
        self._fetch_code_dict()
        bcode = base64.b64encode(bytes(code, "utf-8")) #UTF-8からBase64にエンコード
        nbcode = bcode.decode("utf-8") #Base64からUTF-8にデコード (文字の種類を制限するためにこのような操作？)
        matcher = NodeMatcher(self.graph_db) #matcherの初期化

        if bcode in self.code_dict or nbcode in self.code_dict:
            logging.info("code_dict already has")
            # すでにgraph_dbにcodeが保存されていたとき
            current_cell = matcher.match("Cell", source_code=nbcode).first() # セルのソースコードがBase64の形式で一致するノードを検索
            #current_cell = matcher.match("Cell", source_code=bcode).first() # セルのソースコードがBase64の形式で一致するノードを検索
        else:
            logging.info("code_dict does not have")
            if len(list(self.code_dict.values())) != 0: # code_dictに1つ以上コードが格納されているとき．
                max_id = max(list(self.code_dict.values())) #max_idはcode_dict.values()のうち(おそらくセル番号？)，最も大きい値を返す．
            else:
                max_id = 0
            #current_cell = Node("Cell", name=f"cell_{max_id + 1}", source_code=bcode) #セル番号をインクリメントして，Base64にエンコードされたコードをkeyがsource_codeのvalueに入れる
            current_cell = Node("Cell", name=f"cell_{max_id + 1}", source_code=nbcode) #Base64にエンコードしたコードだとうまくいかない．
            self.graph_db.create(current_cell) #エラー
            self.graph_db.push(current_cell)

            if prev_node is not None:
                # prev_node(ノード)はcurrent_cell(ノード)のSuccessor(辺のラベル)というリレーションを設定, データベースに入れる.
                successor = Relationship(prev_node, "Successor", current_cell)
                self.graph_db.create(successor)
                self.graph_db.push(successor)
                # リレーションをデータベースに入れる. リレーション: current_cellはprev_nodeのParent
                parent = Relationship(current_cell, "Parent", prev_node)
                self.graph_db.create(parent)
                self.graph_db.push(parent)

            self.code_dict[bcode] = max_id + 1 # セル番号のインクリメント
            #self.code_dict[nbcode] = max_id + 1 # セル番号のインクリメント
            try:
                self._store_code_dict() # PostgreSQLでデータベースにインスタンス変数code_dictの内容を保存
            except Exception as e:
                logging.info(f"Code update for Neo4j failed with error {e}")

        if var is None:
            return current_cell
        
        var_name = f"{cell_id}_{var}_{nb_name}" # グラフにおける変数のノード名を設定
        current_var = matcher.match("NotTableVar", name=var_name).first()

        # 各セルに含まれる変数をグラフ形式で保存．
        if current_var is None: #current_varのノードがまだ生成(保存)されていない場合
            current_var = Node("NotTableVar", name=var_name)

            self.graph_db.create(current_var)
            self.graph_db.push(current_var)

            # リレーションを設定． 各セルに対してこの変数が含まれることをリレーションとして設定する．
            contains_edge = Relationship(current_cell, "Contains", current_var)
            self.graph_db.create(contains_edge)
            self.graph_db.push(contains_edge)
            contained_by_edge = Relationship(current_var, "Containedby", current_cell)
            self.graph_db.create(contained_by_edge)
            self.graph_db.push(contained_by_edge)

        return current_cell


    def add_all_cell_and_var_node(self, code_list, prev_node, cell_id, nb_name, var_list2):
        self._fetch_code_dict()
        code="\n".join(code_list[cell_id])
        if code == "":
            #print(f"null cell code: {cell_id}")
            return None
        bcode = base64.b64encode(bytes(code, "utf-8")) #UTF-8からBase64にエンコード
        nbcode = bcode.decode("utf-8") #Base64からUTF-8にデコード (文字の種類を制限するためにこのような操作？)
        matcher = NodeMatcher(self.graph_db) #matcherの初期化

        flg=False
        try: #すでにノードが保存されているか確認
            current_cell = matcher.match("Cell", real_cell_id=cell_id, nb_name=nb_name).first() # セルのソースコードがBase64の形式で一致するノードを検索
            if current_cell is None:
                flg=True
            else:
                pass
                #logging.info(f"cell node already stored into neo4j. nb name: {nb_name}, cell id: {cell_id}")
            # すでにgraph_dbにcodeが保存されていたとき
            #current_cell = matcher.match("Cell", source_code=nbcode).first() # セルのソースコードがBase64の形式で一致するノードを検索
            #current_cell = matcher.match("Cell", source_code=bcode).first() # セルのソースコードがBase64の形式で一致するノードを検索
        except:
            flg=True
        if flg:
            #logging.info("code_dict does not have")
            if len(list(self.code_dict.values())) != 0: # code_dictに1つ以上コードが格納されているとき．
                max_id = max(list(self.code_dict.values())) #max_idはcode_dict.values()のうち(おそらくセル番号？)，最も大きい値を返す．
            else:
                max_id = 0
            #current_cell = Node("Cell", name=f"cell_{max_id + 1}", source_code=bcode) #セル番号をインクリメントして，Base64にエンコードされたコードをkeyがsource_codeのvalueに入れる
            current_cell = Node("Cell", name=f"cell_{max_id + 1}", real_cell_id=cell_id, source_code=nbcode, nb_name=nb_name) #Base64にエンコードしたコードだとうまくいかない．
            #current_cell = Node("Cell", name=f"cell_{max_id + 1}", source_code=nbcode) #Base64にエンコードしたコードだとうまくいかない．
            self.graph_db.create(current_cell) #エラー
            self.graph_db.push(current_cell)

            if prev_node is not None:
                # prev_node(ノード)はcurrent_cell(ノード)のSuccessor(辺のラベル)というリレーションを設定, データベースに入れる.
                successor = Relationship(prev_node, "Successor", current_cell)
                self.graph_db.create(successor)
                self.graph_db.push(successor)
                # リレーションをデータベースに入れる. リレーション: current_cellはprev_nodeのParent
                parent = Relationship(current_cell, "Parent", prev_node)
                self.graph_db.create(parent)
                self.graph_db.push(parent)

            #if bcode not in self.code_dict or nbcode not in self.code_dict:
            self.code_dict[bcode] = max_id + 1 # セル番号のインクリメント
            #self.code_dict[nbcode] = max_id + 1 # セル番号のインクリメント
            try:
                self._store_code_dict() # PostgreSQLでデータベースにインスタンス変数code_dictの内容を保存
            except Exception as e:
                logging.info(f"Code update for Neo4j failed with error {e}")

        if cell_id not in var_list2:
            return current_cell

        for var in var_list2[cell_id][0]: #dependencyの左に格納されている変数たち
            var_name = f"{cell_id}_{var}_{nb_name}" # グラフにおける変数のノード名を設定
            current_var=None
            current_var = matcher.match("AnyTypeVar", name=var_name).first()
            if current_var is None: #current_varのノードがまだ生成(保存)されていない場合
                current_var = Node("AnyTypeVar", name=var_name)
                self.graph_db.create(current_var)
                self.graph_db.push(current_var)
                # リレーションを設定． 各セルに対してこの変数が含まれることをリレーションとして設定する．
                contains_edge = Relationship(current_cell, "Contains", current_var)
                self.graph_db.create(contains_edge)
                self.graph_db.push(contains_edge)
                contained_by_edge = Relationship(current_var, "Containedby", current_cell)
                self.graph_db.create(contained_by_edge)
                self.graph_db.push(contained_by_edge)
        for var2 in var_list2[cell_id][1]: #dependencyの右に格納されている変数たち
            var_name2=None
            for searching_cid in var_list2:
                if searching_cid >= cell_id:
                    break
                if var2 in var_list2[searching_cid][0]:
                    var_name2 = f"{searching_cid}_{var2}_{nb_name}" # グラフにおける変数のノード名を設定
            if var_name2 is None:
                    var_name2 = f"{cell_id}_{var2}_{nb_name}" # グラフにおける変数のノード名を設定
            current_var=None
            current_var = matcher.match("AnyTypeVar", name=var_name2).first()
            if current_var is None: #current_varのノードがまだ生成(保存)されていない場合
                #continue
                current_var = Node("AnyTypeVar", name=var_name2)
                self.graph_db.create(current_var)
                self.graph_db.push(current_var)
            # リレーションを設定． 各セルに対してこの変数が含まれることをリレーションとして設定する．
            contains_edge = Relationship(current_cell, "Uses", current_var)
            self.graph_db.create(contains_edge)
            self.graph_db.push(contains_edge)
            contained_by_edge = Relationship(current_var, "Usedby", current_cell)
            self.graph_db.create(contained_by_edge)
            self.graph_db.push(contained_by_edge)

        return current_cell


    def set_property(self, var, nb_name, cell_id, var_type):
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        var_name = f"{cell_id}_{var}_{nb_name}" # グラフにおける変数のノード名

        current_var = matcher.match("AnyTypeVar", name=var_name).first()
        if current_var is None:
            pass
            #logging.info(f"the Node doesn't exist: nb name {nb_name}, cell id {cell_id}, var name {var}")
        #elif "DataFrame" in var_type or "ndarray" in var_type or "list" in var_type:
        elif "DataFrame" in var_type:
            if "NotTableVar" in current_var.labels:
                current_var.remove_label("NotTableVar")
            if "Module" in current_var.labels:
                current_var.remove_label("Module")
            
            if "Var" in current_var.labels:
                self.graph_db.push(current_var)
                #logging.info(f"""{var} Node label is already \'Var\' and var_type is {current_var['data_type']}""")
            else:
                current_var.add_label("Var")
                current_var["data_type"]=var_type.replace("<class \'","").replace("\'>","").replace("</class>","")
                self.graph_db.push(current_var)
                logging.info(f"changed {var} Node label to \'Var\'") # and set var_type \'{{current_var['data_type']}}\'")
        elif "module" in var_type:
            if "Var" in current_var.labels:
                current_var.remove_label("Var")
            if "NotTableVar" in current_var.labels:
                current_var.remove_label("NotTableVar")
            
            if "Module" in current_var.labels:
                self.graph_db.push(current_var)
                #logging.info(f""" Node label is already \'Module\' and var_type is {current_var['data_type']}""")
            else:
                current_var.add_label("Module")
                current_var["data_type"]=var_type.replace("<class \'","").replace("\'>","").replace("</class>","")
                self.graph_db.push(current_var)
                logging.info(f"changed {var} Node label to \'Module\'")
        else:
            if "Var" in current_var.labels:
                current_var.remove_label("Var")
            if "Module" in current_var.labels:
                current_var.remove_label("Module")
            
            if "NotTableVar" in current_var.labels:
                self.graph_db.push(current_var)
                #logging.info(f""" Node label is already \'NotTableVar\' and var_type is {current_var['data_type']}""")
            else:
                current_var.add_label("NotTableVar")
                current_var["data_type"]=var_type.replace("<class \'","").replace("\'>","").replace("</class>","")
                self.graph_db.push(current_var)
                logging.info(f"changed \'{var}\' Node label to \'NotTableVar\'")

    def set_display_property(self, display_data, nb_name, cell_id):
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
        node_name = f"{cell_id}_displaydata_{nb_name}"
        current_cell = matcher.match("Cell", real_cell_id=cell_id, nb_name=nb_name).first()
        if not current_cell is None:
            try:
                node_display=None
                
                node_display = matcher.match("Display_data", name=node_name,data_type=display_data).first()
                if node_display is None:
                    node_display = Node("Display_data", name=node_name, data_type=display_data, cell_id=cell_id, nb_name=nb_name) 
                    self.graph_db.create(node_display)
                    self.graph_db.push(node_display)
                
                display_edge=r_matcher.match((current_cell, node_display), "Display").first()
                if display_edge is None:
                    display_edge = Relationship(current_cell, "Display", node_display)
                    self.graph_db.create(display_edge)
                    self.graph_db.push(display_edge)

                displayedby_edge=r_matcher.match((node_display,current_cell), "Displayedby").first()
                if displayedby_edge is None:
                    displayedby_edge = Relationship(node_display, "Displayedby", current_cell)
                    self.graph_db.create(displayedby_edge)
                    self.graph_db.push(displayedby_edge)
            except:
                logging.info(f"setting display '{node_name}' failed")
                pass
        else:
            logging.info(f"storing outputs node failed: not found parent node of cell")
            self.current_cell2 = matcher.match("Cell", nb_name=nb_name).all()
            
    def reset_display_property_1(self, display_data, nb_name, cell_id):
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        r_matcher = RelationshipMatcher(self.graph_db) #matcherの初期化
        old_node_name = f"{cell_id}_displaydata_{nb_name}"
        node_name = f"{cell_id}_{display_data}_{nb_name}"
        current_cell = matcher.match("Cell", real_cell_id=cell_id, nb_name=nb_name).first()
        if not current_cell is None:
            try:
                node_display=None
                
                node_display = matcher.match("Display_data", name=old_node_name, cell_id=cell_id, data_type=display_data).first()
                if not node_display is None:
                    node_display["name"]=node_name
                    self.graph_db.push(node_display)
                else:
                    node_display=None
                    node_display = matcher.match("Display_data", name=node_name, cell_id=cell_id, data_type=display_data).first()
                    if node_display is None:
                        node_display = Node("Display_data", name=node_name, data_type=display_data, cell_id=cell_id, nb_name=nb_name) 
                        self.graph_db.create(node_display)
                        self.graph_db.push(node_display)
                    
                    display_edge=r_matcher.match((current_cell, node_display), "Display").first()
                    if display_edge is None:
                        display_edge = Relationship(current_cell, "Display", node_display)
                        self.graph_db.create(display_edge)
                        self.graph_db.push(display_edge)

                    displayedby_edge=r_matcher.match((node_display,current_cell), "Displayedby").first()
                    if displayedby_edge is None:
                        displayedby_edge = Relationship(node_display, "Displayedby", current_cell)
                        self.graph_db.create(displayedby_edge)
                        self.graph_db.push(displayedby_edge)
            except:
                logging.info(f"setting display '{node_name}' failed")
                pass
        else:
            logging.info(f"storing outputs node failed: not found parent node of cell")
            self.current_cell2 = matcher.match("Cell", nb_name=nb_name).all()
            
        



    def store_libraries(self, import_lib, nb_name):
        """
        1つのNBでインポートされている全てのライブラリをpostgresqlでDBに格納する．

        Args:
            import_lib (list[str]): NBにインポートされたライブラリのリスト
            nb_name (str): cleaned nb name.
        """
        conn=self.postgres_eng
        lib_dict={}
        try:
            lib_table = pd.read_sql_table(
                "nb_libraries", conn, schema=config.sql.nb_info
            )
        except:
            pass
        
        for index, row in lib_table.iterrows():
            if row["nb_name"] not in lib_dict:
                lib_dict[row["nb_name"]] = []
            lib_dict[row["nb_name"]].append(row["libraries"])
        for lib in import_lib:
            if not (nb_name in lib_dict and lib in lib_dict[nb_name]):
                conn.execute(
                    f"INSERT INTO {config.sql.nb_info}.nb_libraries VALUES ('{nb_name}', '{lib}');"
                )
                #conn.execute(
                    #f"INSERT INTO {config.sql.nb_info}.nb_libraries(nb_name, libraries) SELECT '{nb_name}', '{lib}' WHERE NOT EXISTS (SELECT {nb_name}, {lib} FROM {config.sql.nb_info}.nb_libraries WHERE nb_name='{nb_name}', libraries='{lib}'');"
                #)
        logging.info(f"inserting imported libraries in NB {nb_name} to DB: complete.")


    def store_nottablevar(self, store_name, cell_id, var_name, var_contents):
        """
        1つのNBでインポートされている全てのライブラリをpostgresqlでDBに格納する．

        Args:
            import_lib (list[str]): NBにインポートされたライブラリのリスト
            nb_name (str): cleaned nb name.
        """
        conn=self.postgres_eng
        query2 = "CREATE TABLE IF NOT EXISTS " + config.sql.nottablevar + "."+store_name+" (cell_id INTEGER, var_name VARCHAR(1000), var_contents VARCHAR(100000));"
        dict={}
        with self.postgres_eng.connect() as conn:
            try:
                conn.execute(query2)
            except Exception as e:
                logging.error(
                    f"Store NotTableVar contents: dropping of table failed with error {e}"
                )
        try:
            old_table = pd.read_sql_table(
                store_name, conn, schema=config.sql.nottablevar
            )
        except:
            pass
        
        for index, row in old_table.iterrows():
            if row["cell_id"] not in dict:
                dict[row["cell_id"]] = []
            dict[row["cell_id"]].append(row["var_name"])
        if not (cell_id in dict and var_name in dict[cell_id]):
            try:
                var_contents = json.dumps(var_contents)
            except:
                pass
            try:
                var_contents = str(var_contents)
            except:
                pass
            try:
                conn.execute(
                    f"INSERT INTO {config.sql.nottablevar}.{store_name} VALUES ('{cell_id}', '{var_name}', '{var_contents}');"
                )
                logging.info(f"storing var contents completed : store name {store_name}, cell id {cell_id}, var name{var_name}")
            except:
                logging.info(f"storing var contents failed : store name {store_name}, cell id {cell_id}, var name{var_name}")

    def gather_table_var(self, cleaned_nb_name):
        matcher = NodeMatcher(self.graph_db) #matcherの初期化
        var_name_list = matcher.match("Var").all()
        ret_list=[]
        store_name_dict={}
        for node in var_name_list:
            if cleaned_nb_name in node["name"]:
                #var="_".join((var.split("_"))[2:])
                var=node["name"]
                var=var[var.index("_")+1:]
                var=var[:var.index("_")]
                ret_list.append(var)
                store_name_dict[var]=node["name"]
        return list(set(ret_list)), store_name_dict


    def insert_display_data_type(self, store_name, display_type_list):
        # store name: cleaned nb name
        #logging.info("storing display data type...")
        type_dict={}
        with self.postgres_eng.connect() as conn:
            try:
                type_db = pd.read_sql_table("display_type", conn, schema=config.sql.nb_info)
                for _, d in type_db.iterrows():
                    if d["nb_name"] == store_name:
                        if d["data_type"] not in type_dict:
                            type_dict[d["cell_id"]]=[]
                        type_dict[d["cell_id"]].append(d["data_type"])
                #var_list = type_db["view_id"].tolist() # ".tolist()": DataFrameをlistに変換
            except Exception as e:
                logging.error(f"Reading display data type from database failed due to error {e}")

        for cell_id in display_type_list:
            for data_type in display_type_list[cell_id]:
                if cell_id in type_dict and data_type in type_dict[cell_id]:
                    #print(f"already stored: {cell_id}, {data_type}")
                    continue
                try:
                    with self.postgres_eng.connect() as conn:
                        conn.execute(
                            f"INSERT INTO {config.sql.nb_info}.display_type VALUES ('{store_name}', '{cell_id}', '{data_type}')"
                        )
                except Exception as e:
                    logging.error(f"Storing display data type to database failed due to error {e}")




    def fetch_and_set_display_data_type(self):
        with self.postgres_eng.connect() as conn:
            try:
                display_data_table = pd.read_sql_table("display_type", conn, schema=f"{config.sql.nb_info}")
                for _, row in display_data_table.iterrows():
                    display_data=row["data_type"]
                    nb_name=row["nb_name"]
                    cell_id=row["cell_id"]
                    #print((display_data, nb_name, cell_id))
                    self.set_display_property(display_data, nb_name, cell_id)
            except:
                pass