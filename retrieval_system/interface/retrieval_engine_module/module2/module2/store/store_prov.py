import ast
import json
import logging

import networkx as nx
import pandas as pd
import psycopg2

from module2.config import config
from module2.utils.funclister import FuncLister

special_type = ["np", "pd"]


class LineageStorage:
    """
    table名は"dependen", "line2cid", "lastliid"
    テーブルのスキーマは(view_id, view_cmd) : 関数`build_table`参照

    self.name str:
        eng: PostgreSQLのインスタンス(エンジン).
        variable (list): store_nameのリスト．(関数insert_table_model参照)
        view_cmd (dict): __parse_codeの戻り値のうち，dependencyを格納．キーはstore_name．
        l2d_cmd (dict): __parse_codeの戻り値のうち，line2cidを格納．キーはstore_name．
    """
    def __init__(self, psql_eng):
        """
        Args:
            psql_eng: PostgreSQLのインスタンス．
        """
        self.eng = psql_eng
        self.__connect2db()
        self.variable = [] 
        self.view_cmd = {}
        self.l2d_cmd = {}

    @staticmethod
    def __connect2db():
        """
        データベースに名前が`dependen`, `line2cid`, `lastliid`のテーブルが存在しなければ作る．
        テーブルのスキーマは(view_id, view_cmd)
        """
        conn_string = ( # 名前やパスワードなどの各種文字列を一括で管理
            f"host='{config.sql.host}' dbname='{config.sql.dbname}' "
            f"user='{config.sql.name}' password='{config.sql.password}'"
        )

        try:
            # conn.cursor will return a cursor object, you
            # can use this cursor to perform queries.
            conn = psycopg2.connect(conn_string)
            logging.info("Connection to database successful")
            cursor = conn.cursor()

            def build_table(name):
                """
                SQLの操作の文字列．
                データベースに名前が`name`のテーブルが存在しなければ作る．
                テーブルのスキーマは(view_id, view_cmd)
                """
                return (
                    f"CREATE TABLE IF NOT EXISTS {config.sql.graph}.{name} " 
                    f"(view_id VARCHAR(1000), view_cmd VARCHAR(10000000));" # VARCHAR(M)は, 最大M文字数の可変長文字列の型.
                )

            tables = ["dependen", "line2cid", "lastliid"]
            try:
                for table in tables:
                    cursor.execute(build_table(table))
                    conn.commit() # 保存
            except Exception as e:
                logging.error(f"Creation of tables failed due to error {e}")

            cursor.close()
            conn.close()
        except Exception as e:
            logging.error(f"Connection to database failed due to error {e}")

    @staticmethod
    def __last_line_var(varname, code):
        """
        varnameで指定する変数名がcodeに最後に代入などの操作をされる行(varname=operationの形の行)の行番号を返す．

        Args:
            varname (str): 探したい変数名．
            code (str): 検索対象のソースコードの文字列．
        
        Returns:
            int: `varname`が`code`内部で最後に操作される行番号．
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
                    ret = id + 1
        return ret

    @staticmethod
    def __parse_code(code_list):
        """
        コードを整形し抽象度の高いASTにする．

        Args:
            code_list (list): セルのコードをセルごとに切って各添字に格納した配列．

        Returns:
            dict {[str], [str, str]}: FuncLister.dependency
            dict [int: int]: 全てのセルを統合したコードリストの行番号に対し,セルIDとの対応関係を格納したもの．[セルを統合したあとのコードの行番号: セルID]
            str : コード全体の文字列．
        """
        test = FuncLister()
        all_code = ""
        line2cid = {}

        lid = 1
        fflg = False
        for cid, cell in enumerate(code_list):
            if "\\n" in cell:
                codes = cell.split("\\n")
            elif "\n" in cell:
                codes = cell.split("\n")
            else: # 追記(無いとエラーがでたため) cellの型がリストのとき(つまり，もとから行ごとにsplitされていたとき)
                codes = cell
            new_codes = []
            for code in codes:
                if code == "":
                    continue

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

                if not code or code[0] in ["%", "#", " ", "\n"]:
                    continue

                try:
                    ast.parse(code)
                    if not fflg:
                        new_codes.append(code)
                        line2cid[lid] = cid
                        lid = lid + 1
                except Exception as e:
                    logging.info(
                        f"Parsing error in code fragment {code} due to error {e}"
                    )

            all_code = all_code + "\n".join(new_codes) + "\n"

        all_code = all_code.strip("\n")
        all_code = all_code.split("\n")
        all_code = [t for t in all_code if t != ""]
        all_code = "\n".join(all_code)

        tree = ast.parse(all_code)
        test.visit(tree)
        return test.dependency, line2cid, all_code

    def parse_code2(self, code_list):
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

    def generate_graph(self, code_list, nb_name):
        """
        コードからグラフを生成する．グラフは，NetworkXを利用した有向グラフ(DiGraph)である．

        Args:
            code_list (list): セルのコードをセルごとに切って各添字に格納した配列．
            nb_name (str): 整形したあとのノートブック名．グラフのノード名を決める時とに利用．
        Returns:
            DiGraph: 入力をDAGに変換したもの．
            dist[int][int]: セルのコードに対応する行番号を, カラの行やdefを考慮して番号1から順に詰めて整理したもの．[変換後の番号: 元の行番号]
        """
        dependency, line2cid, all_code = self.__parse_code(code_list)
        G = nx.DiGraph()
        for i in dependency.keys():
            left = dependency[i][0] #元のセルにおいてその行が書かれていた行番号が格納.
            right = list(set(dependency[i][1])) #dependency[i][1]はlist[変数名, list[辺で接続先のノード, 辺のラベル]].  e.g. 関数add(int, int)を定義してc=add(a,b) -> dependency[i][1]は ['c'], [('a', 'add'), ('b', 'add')]

            left_node = []
            for ele in left:
                if type(ele) is tuple:
                    ele = ele[0]
                left_node.append(f"var_{ele}_{i}_{nb_name}")

            for ele in left:
                if type(ele) is tuple:
                    ele = ele[0]

                new_node = f"var_{ele}_{i}_{nb_name}"
                G.add_node(new_node, cell_id=line2cid[i], line_id=i, var=ele)

                for dep, ename in right:
                    candidate_list = G.nodes
                    rankbyline = []
                    for cand in candidate_list:
                        if G.nodes[cand]["var"] == dep:
                            if cand in left_node:
                                continue
                            rankbyline.append((cand, G.nodes[cand]["line_id"]))
                    rankbyline = sorted(rankbyline, key=lambda d: d[1], reverse=True)

                    if len(rankbyline) == 0:
                        if dep not in special_type:
                            candidate_node = (
                                "var_" + dep + "_" + str(1) + "_" + str(nb_name)
                            )
                            G.add_node(candidate_node, cell_id=0, line_id=1, var=dep)
                        else:
                            candidate_node = dep + str(nb_name)
                            G.add_node(candidate_node, cell_id=0, line_id=1, var=dep)

                    else:
                        candidate_node = rankbyline[0][0]

                    if dep in special_type:
                        ename = dep + "." + ename
                        G.add_edge(new_node, candidate_node, label=ename)
                    else:
                        G.add_edge(new_node, candidate_node, label=ename)

        return G, line2cid

    def insert_table_model(self, store_name, var_name, code_list):
        """
        code_listを関数__parse_code()に渡して得られる3つを，
        テーブル"dependen", "line2cid", "lastliid"にそれぞれINSERTする．

        Args:
            store_name (str): テーブルを保存するときの名前．
            var_name (str): テーブルを保持している変数名．つまり型がDataFrame, ndarray, listのいずれかの変数．
            code_list (list[str]): セルのソースコードをセルで分割してリスト型に格納したノートブック全体のコードのリスト．

        dep_db (DataFrame): テーブル"dependen"の内容．

        var_list (list): テーブル"dependen"のうち，キーが"view_id"の行のリスト．
        """
        logging.info("Updating provenance...")
        with self.eng.connect() as conn:
            try:
                dep_db = pd.read_sql_table("dependen", conn, schema=config.sql.graph)
                var_list = dep_db["view_id"].tolist() # ".tolist()": DataFrameをlistに変換
            except Exception as e:
                logging.error(f"Reading prov from database failed due to error {e}")

        try:
            logging.error(f"getting  dep, c2i, all_code ...")
            dep, c2i, all_code = self.__parse_code2(code_list)
            logging.error(f"getting  lid ...")
            lid = self.__last_line_var(var_name, all_code)
        except Exception as e:
            logging.error(f"Parse code failed due to error {e}")
            return

        try:
            dep_str = json.dumps(dep)
            l2c_str = json.dumps(c2i)
            lid_str = json.dumps(lid)

            logging.info("JSON created")

            self.view_cmd[store_name] = dep_str
            self.l2d_cmd[store_name] = l2c_str

            encode1 = dep_str
            encode2 = l2c_str

            if (store_name not in var_list) and (store_name not in self.variable):
                logging.debug("Inserting values into dependen and line2cid")

                def insert_value(table_name, encode):
                    """
                    データベースのテーブル"table_name"に(store_name, encode)のタプルをINSERTする．
                    """
                    conn.execute(
                        f"INSERT INTO {config.sql.graph}.{table_name} VALUES ('{store_name}', '{encode}')"
                    )

                with self.eng.connect() as conn: # データベースの各テーブルにそれぞれ対応する内容を追加．
                    try:
                        for table, encoded in [
                            ("dependen", encode1), #encode1 = dep_str
                            ("line2cid", encode2), #encode2 = l2c_str
                            ("lastliid", lid_str),
                        ]:
                            insert_value(table, encoded)
                        self.variable.append(store_name)
                    except Exception as e:
                        logging.error(f"Unable to insert into tables due to error {e}")

        except Exception as e:
            logging.error(f"Unable to update provenance due to error {e}")
            
    def insert_table_model2(self, store_name, var_name, dep, c2i, all_code):
        """
        テーブル"dependen", "line2cid", "lastliid"にそれぞれINSERTする．
        """
        logging.info("Updating provenance...")
        with self.eng.connect() as conn:
            try:
                dep_db = pd.read_sql_table("dependen", conn, schema=config.sql.graph)
                var_list = dep_db["view_id"].tolist() # ".tolist()": DataFrameをlistに変換
            except Exception as e:
                logging.error(f"Reading prov from database failed due to error {e}")

        try:
            lid = self.__last_line_var(var_name, all_code)
        except Exception as e:
            logging.error(f"Parse code failed due to error {e}")
            return

        try:
            dep_str = json.dumps(dep)
            l2c_str = json.dumps(c2i)
            lid_str = json.dumps(lid)

            self.view_cmd[store_name] = dep_str
            self.l2d_cmd[store_name] = l2c_str

            encode1 = dep_str
            encode2 = l2c_str

            if (store_name not in var_list) and (store_name not in self.variable):
                logging.debug("Inserting values into dependen and line2cid")

                def insert_value(table_name, encode):
                    """
                    データベースのテーブル"table_name"に(store_name, encode)のタプルをINSERTする．
                    """
                    conn.execute(
                        f"INSERT INTO {config.sql.graph}.{table_name} VALUES ('{store_name}', '{encode}')"
                    )

                with self.eng.connect() as conn: # データベースの各テーブルにそれぞれ対応する内容を追加．
                    try:
                        for table, encoded in [
                            ("dependen", encode1), #encode1 = dep_str
                            ("line2cid", encode2), #encode2 = l2c_str
                            ("lastliid", lid_str),
                        ]:
                            insert_value(table, encoded)
                        self.variable.append(store_name)
                    except Exception as e:
                        logging.error(f"Unable to insert into tables due to error {e}")

        except Exception as e:
            logging.error(f"Unable to update provenance due to error {e}")
            

    def close_dbconnection(self):
        self.eng.close()
