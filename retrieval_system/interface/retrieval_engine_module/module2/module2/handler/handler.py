import json
import logging

import pandas as pd
from notebook.base.handlers import IPythonHandler

from module2.config import config
from module2.db.table_db import connect2db_engine, connect2gdb
from module2.jupyter import jupyter
from module2.search.search import search_tables
from module2.store.store_graph import ProvenanceStorage
from module2.store.store_prov import LineageStorage
from module2.utils.utils import clean_notebook_name


class JuneauHandler(IPythonHandler):
    """
    Juneau Handlerはノートブックサーバーアプリケーションのインスタンスを調整します．
    基本的には，このクラスはフロントエンドとバックエンド間で，PUTとPOSTを介して通信するという役割です．
    このserver_extension.pyが呼び出されます．
    IPythonHandlerについて: "Registering custom handlers" https://jupyter-notebook.readthedocs.io/en/stable/extending/handlers.html
    """

    def initialize(self):
        """
        Initializes all the metadata related to a table in a Jupyter Notebook.
        Note we use `initialize()` instead of `__init__()` as per Tornado's docs:
        https://www.tornadoweb.org/en/stable/web.html#request-handlers
        Jupyter Notebok内の表に関わる全てのメタデータを初期化します．
        `__init__()`の代わりに`initialize()`を使っていることに注意してください．(Tornadoのドキュメントを参照．)

        The metadata related to the table is:
            - var: the name of the variable that holds the table. テーブルを保持している変数名．
            - kernel_id: the id of the kernel that executed the table. テーブルが実行された時のカーネルID(jupyterごとのID)．
            - cell_id: the id of the cell that created the table. テーブルが作られた時のセルID．
            - code: the actual code associated to creating the table. テーブルの作成に関連づけられた実際のコード．
            - mode: モードの指定．各モードに対応した内容は以下の通り．(juneau/search/search.py参照)
                1. Searching for additional training data. 
                2. Searching for joinable tables.
                3. Searching for alternative features.
            - nb_name: the name of the notebook. ノートブック名．
            - done: TODO 接続しているカーネルIDの重複無しリスト(set)．
            - data_trans: TODO
            - graph_db: the Neo4J graph instance. Neo4JのGraphインスタンス．
            - psql_engine: Postgresql engine.
            - store_graph_db_class: TODO
            - store_prov_db_class: LineageStorageのインスタンス．(store_prov.py参照)
            - prev_node: TODO

        Global application variables: (server_extension.py参照)
            self.application.indexed (set{str}): データベースにストア済みのテーブル名．
            self.application.nb_cell_id_node (dict{str: {int: Node}}): それぞれセルIDとノードを格納し，dict{nb_name: {cell_id: Node}}．
            self.application.search_test_class (WithProv_Optimized) # search_test_classはserver_extension.pyでWithProv_Optimized(config.sql.dbname, config.sql.dbs)で指定されている．

        Notes:
            Depending on the type of request (PUT/POST), some of the above will
            be present or not. For instance, the notebook name will be present
            on PUT but not on POST. That is why we check if the key is present in the
            dictionary and otherwise assign it to `None`.

            This function is called on *every* request.

        """
        data = self.request.arguments
        self.var = data["var"][0].decode("utf-8")
        self.kernel_id = data["kid"][0].decode("utf-8")
        self.code = data["code"][0].decode("utf-8")
        self.cell_id = (
            int(data["cell_id"][0].decode("utf-8")) if "cell_id" in data else None
        )
        self.mode = int(data["mode"][0].decode("utf-8")) if "mode" in data else None
        self.nb_name = data["nb_name"][0].decode("utf-8") if "nb_name" in data else None

        self.done = set()
        self.data_trans = {}
        self.graph_db = None
        self.psql_engine = None
        self.store_graph_db_class = None
        self.store_prov_db_class = None
        self.prev_node = None

    def find_variable(self):
        """
        Finds and tries to return the contents of a variable in the notebook.
        ノートブック内の変数self.varの中身を探し、
        self.varの型がDataFrame, ndarray, listのいずれかであればpandas DataFrame形式でvar_objを返す．
        上記以外の型ならばエラーを返す．

        Returns:
            tuple - the status (`True` or `False`), and the variable if `True`.
        """
        # Make sure we have an engine connection in case we want to read.
        if self.kernel_id not in self.done: # まだこのカーネルIDのカーネルに接続していないとき．
            o2, err = jupyter.exec_connection_to_psql(self.kernel_id)
            self.done.add(self.kernel_id)
            logging.info(o2)
            logging.info(err)

        logging.info(f"Looking up variable {self.var}")
        output, error = jupyter.request_var(self.kernel_id, self.var) # 表データがある場合, outputにjson形式で格納する.
        logging.info("Returned with variable value.")

        if error or not output: # 表データが無い場合
            sta = False
            return sta, error
        else:
            try:
                var_obj = pd.read_json(output, orient="split")
                sta = True
            except Exception as e:
                logging.error(f"Found error {e}")
                var_obj = None
                sta = False

        return sta, var_obj

    def put(self):
        """
        実行したセルのセルとテーブルをデータベースにStoreする．
        セルはノード間のリレーションを設定する事により，ワークフローにノードとして追加することを実現している．

        self.prev_node (Node): 次にセルをノードとして追加するときにリレーションで接続するノード．
        """
        logging.info(f"Juneau indexing request: {self.var}")
        logging.info(f"Stored tables: {self.application.indexed}")

        cleaned_nb_name = clean_notebook_name(self.nb_name) #ノートブック名を整える
        code_list = self.code.strip("\\n#\\n").split("\\n#\\n") #codeを改行で区切ってリストにする
        store_table_name = f"{self.cell_id}_{self.var}_{cleaned_nb_name}" #テーブルを保存する時の名前を自動で生成

        if store_table_name in self.application.indexed: #すでに同じ名前のテーブル名が存在する時
            logging.info("Request to index is already registered.")
        elif self.var not in code_list[-1]:
            logging.info("Not a variable in the current cell.")
        else: # セルをワークフローグラフに保存. テーブルもデータベースに保存.
            logging.info(f"Starting to store {self.var}")
            success, output = self.find_variable()

            if success:
                logging.info(f"Getting value of {self.var}")
                logging.info(output.head())

                # Noneのときそれぞれの変数に対応するインスタンスを設定．
                if not self.graph_db:
                    self.graph_db = connect2gdb()
                if not self.psql_engine:
                    self.psql_engine = connect2db_engine(config.sql.dbname)
                if not self.store_graph_db_class:
                    self.store_graph_db_class = ProvenanceStorage(
                        self.psql_engine, self.graph_db
                    )
                if not self.store_prov_db_class:
                    self.store_prov_db_class = LineageStorage(self.psql_engine)

                if cleaned_nb_name not in self.application.nb_cell_id_node:
                    self.application.nb_cell_id_node[cleaned_nb_name] = {}

                # セルをデータベースに保存(ワークフローグラフへの追加)
                try:
                    # cell_id以下の番号のうち、nb_cell_id_nodeに格納されている最大のcell_idをもつノードをprev_nodeとする. なければNoneのまま
                    for cid in range(self.cell_id - 1, -1, -1): # C++でいうfor(int i=cell_id; i>-1; i--)と同じ
                        if cid in self.application.nb_cell_id_node[cleaned_nb_name]:
                            self.prev_node = self.application.nb_cell_id_node[cleaned_nb_name][cid]
                            break
                    self.prev_node = self.store_graph_db_class.add_cell( # for cell provenance tracking (セルをワークフローグラフに追加)
                        self.code,
                        self.prev_node,
                        self.var,
                        self.cell_id,
                        cleaned_nb_name,
                    )
                    # cell_idがまだnb_cell_id_nodeに格納されていなければ，その添字cell_idにprev_nodeを格納する
                    if self.cell_id not in self.application.nb_cell_id_node[cleaned_nb_name]:
                        self.application.nb_cell_id_node[cleaned_nb_name][self.cell_id] = self.prev_node
                except Exception as e:
                    logging.error(f"Unable to store in graph store due to error {e}")

                self.store_table(output, store_table_name) # テーブルをデータベースに保存

            else:
                logging.error("find variable failed!")

        self.data_trans = {"res": "", "state": str("true")}
        self.write(json.dumps(self.data_trans)) #Jupyterに表示

    def post(self):
        """
        テーブルを検索して結果をブラウザ(Jupyter Notebook)上に表示する．
        """
        logging.info("Juneau handling search request")
        if self.mode == 0:  # return table
            self.data_trans = {
                "res": "",
                "state": self.var in self.application.search_test_class.real_tables, # search_test_classはserver_extension.pyでWithProv_Optimized(config.sql.dbname, config.sql.dbs)で指定されている．
            }
            self.write(json.dumps(self.data_trans))
        else:
            success, output = self.find_variable() # outputはNoneまたはDataFrame
            if success:
                # テーブルを検索
                data_json = search_tables( 
                    self.application.search_test_class, output, self.mode, self.code, self.var
                )
                self.data_trans = {"res": data_json, "state": data_json != ""}
                self.write(json.dumps(self.data_trans))
            else:
                logging.error(f"The table was not found: {output}")
                self.data_trans = {"error": output, "state": False}
                self.write(json.dumps(self.data_trans))

    def store_table(self, output, store_table_name):
        """
        Asynchronously stores a table into the database.
        テーブルを非同期でPostgreSQLでデータベースにストアし，
        セルのコードもストア．

        Args:
            output (DataFrame, ndarray, or list): ストアしたい表データ．
            store_table_name (str): ストアするテーブル名．

        Notes:
            This is the refactored version of `fn`.

        """
        logging.info(f"Indexing new table {store_table_name}")
        conn = self.psql_engine.connect()

        try:
            output.to_sql(
                name=f"rtable{store_table_name}",
                con=conn,
                schema=config.sql.dbs, # schema name(not table definitions)
                if_exists="replace", # 既にあった場合は置き換え
                index=False,
            )
            logging.info("Base table stored")
            try:
                code_list = self.code.split("\\n#\\n")
                self.store_prov_db_class.insert_table_model(
                    store_table_name, self.var, code_list
                )
                self.application.indexed.add(store_table_name)
            except Exception as e:
                logging.error(
                    f"Unable to store provenance of {store_table_name} "
                    f"due to error {e}"
                )
            logging.info(f"Returning after indexing {store_table_name}")
        except Exception as e:
            logging.error(f"Unable to store {store_table_name} due to error {e}")
        finally:
            conn.close()
