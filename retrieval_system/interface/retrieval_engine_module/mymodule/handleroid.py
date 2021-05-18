#handler.pyを調整

import json
import logging

import pandas as pd
from notebook.base.handlers import IPythonHandler
import numpy as np

from juneau.config import config
from juneau.db.table_db import connect2db_engine, connect2gdb
from juneau.jupyter import jupyter
from juneau.search.search import search_tables
from juneau.store.store_graph import ProvenanceStorage
from juneau.store.store_prov import LineageStorage
from juneau.utils.utils import clean_notebook_name


class Handleroid():
    
    def __init__(self, kernel_id, nb_name, nb_all_code):
        self.var = None
        self.kernel_id = kernel_id
        self.code = None
        self.nb_all_code=nb_all_code #てのセルを結合したコードを1行ごとに全区切ってリスト化したノートブック全体のコード．
        self.cell_id = None
        self.nb_name = nb_name

        self.done = set()
        self.data_trans = {}
        self.graph_db = connect2gdb()
        self.psql_engine = connect2db_engine(config.sql.dbname)
        self.store_graph_db_class = ProvenanceStorage(self.psql_engine, self.graph_db)
        self.store_prov_db_class = LineageStorage(self.psql_engine)
        self.prev_node = None
        self.nb_cell_id_node = {}
        self.indexed=set()
        self.init_schema()
        self.store_graph_db_class = ProvenanceStorage(self.psql_engine, self.graph_db)

        #for i in range(len(self.nb_all_code)):
        #    if type(self.nb_all_code[i]) is type([]):
        #        self.nb_all_code[i]="\n".join(self.nb_all_code[i])

    #不使用
    def find_variable(self):
        """
        Finds and tries to return the contents of a variable in the notebook.
        ノートブック内の変数self.varの中身を探し、
        self.varの型がDataFrame, ndarray, listのいずれかであればjson形式で返す．
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

        #if error or not output or (var_type != "table"): # 表データが無い場合
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

    def store_all_cell_and_var(self, var_outputs, cell_code_list):
        cleaned_nb_name = clean_notebook_name(self.nb_name) #ノートブック名を整える
        store_cell_name=f"cellcont_{cleaned_nb_name}" # DBのスキーマnb_cellcodeに入れる時のテーブル名
        try: # (cell_id, cell_code)のタプルをデータベースのcolumsとして初期化
            self.init_table(store_cell_name)
        except Exception as e:
            logging.error(f"init_table: error {e}")   

        for cell_id in cell_code_list:
            if self.nb_all_code[cell_id] == "": #コードの内容がカラの場合(当然そのセルには変数も登場しない)
                continue
            #self.store_cell(cell_id, store_cell_name, cell_code_list[cell_id]) #セルそのままの内容をデータベースへ追加

            #以下で変数の内容を保存

            
            if cell_id not in var_outputs:
                continue
            for var_name in var_outputs[cell_id][0]:
                store_var_name = f"{cell_id}_{var_name}_{cleaned_nb_name}" #テーブルを保存する時の名前を生成
                try: #変数の内容をデータベースに保存
                    var_obj=pd.read_json(var_outputs[cell_id][0][var_name][1]["text"], orient="split")
                    #logging.info("storing DataFrame to DB ...")
                    #*************************ここまでいけてる
                    self.store_table2(var_name, var_obj, store_var_name)
                    logging.info(f"table \'{var_name}\' stored.")
                    # print(f"""table {var_outputs[cell_id][var_name]} stored.""")
                except:
                    var_obj=var_outputs[cell_id][0]
                    #logging.info(f"var \'{var_name}\' is not table type.")
                    #try:
                    #    self.store_graph_db_class.store_nottablevar(store_cell_name, cell_id, var_name, var_outputs[cell_id][0][var_name][1])
                    #except:
                    #    logging.info(f"failed: store \'{var_name}\' contents")

            for var_name in var_outputs[cell_id][1]:
                store_var_name = f"{cell_id}_{var_name}_{cleaned_nb_name}" #テーブルを保存する時の名前を生成
                try: #変数の内容をデータベースに保存
                    #*************************ここまでいけてる
                    var_obj=pd.read_json(var_outputs[cell_id][1][var_name][1]["text"], orient="split")
                    #logging.info("storing DataFrame to DB ...")
                    self.store_table2(var_name, var_obj, store_var_name)
                    logging.info(f"table \'{var_name}\' stored.")
                except:
                    var_obj=var_outputs[cell_id][1]
                    #logging.info(f"var \'{var_name}\' is not table type.")
                    #try:
                    #    self.store_graph_db_class.store_nottablevar(cleaned_nb_name, cell_id, var_name, var_outputs[cell_id][0][var_name][1])
                    #except:
                    #    logging.info(f"failed: store \'{var_name}\' contents")
            #print(i)
    def store_all_var_new(self, var_outputs, cell_code_list):
        cleaned_nb_name = clean_notebook_name(self.nb_name) #ノートブック名を整える
        for cell_id in cell_code_list:
            if self.nb_all_code[cell_id] == "": #コードの内容がカラの場合(当然そのセルには変数も登場しない)
                continue
            #以下で変数の内容を保存
            if cell_id not in var_outputs:
                continue
            for var_name in var_outputs[cell_id][0]:
                store_var_name = f"{cell_id}_{var_name}_{cleaned_nb_name}" #テーブルを保存する時の名前を生成
                try: #変数の内容をデータベースに保存
                    var_obj=pd.read_json(var_outputs[cell_id][0][var_name][1]["text"], orient="split")
                    self.store_table2(var_name, var_obj, store_var_name)
                    logging.info(f"table \'{var_name}\' stored.")
                except:
                    if "DataFrame" in var_outputs[cell_id][0][var_name][0] or "ndarray" in var_outputs[cell_id][0][var_name][0]:
                        logging.info(f"failed storing table: {store_var_name}")
                        ret_err=True
            for var_name in var_outputs[cell_id][1]:
                store_var_name = f"{cell_id}_{var_name}_{cleaned_nb_name}" #テーブルを保存する時の名前を生成
                try: #変数の内容をデータベースに保存
                    var_obj=pd.read_json(var_outputs[cell_id][1][var_name][1]["text"], orient="split")
                    self.store_table2(var_name, var_obj, store_var_name)
                    logging.info(f"table \'{var_name}\' stored.")
                except:
                    if "DataFrame" in var_outputs[cell_id][1][var_name][0] or "ndarray" in var_outputs[cell_id][1][var_name][0]:
                        logging.info(f"failed storing table: {store_var_name}")
                        ret_err=True
                    pass

    def store_all_var(self, var_outputs, cell_code_list):
        cleaned_nb_name = clean_notebook_name(self.nb_name) #ノートブック名を整える
        store_cell_name=f"cellcont_{cleaned_nb_name}" # DBのスキーマnb_cellcodeに入れる時のテーブル名

        ret_err=False   

        for cell_id in cell_code_list:
            if self.nb_all_code[cell_id] == "": #コードの内容がカラの場合(当然そのセルには変数も登場しない)
                continue

            #以下で変数の内容を保存
            if cell_id not in var_outputs:
                continue

            for var_name in var_outputs[cell_id][0]:
                store_var_name = f"{cell_id}_{var_name}_{cleaned_nb_name}" #テーブルを保存する時の名前を生成
                try: #変数の内容をデータベースに保存
                    var_obj=pd.read_json(var_outputs[cell_id][0][var_name][1]["text"], orient="split")
                    try:
                        for col in var_obj.columns:
                            if "(" in col:
                                col_new=col.replace("(", "")
                                while "(" in col_new:
                                    col_new=col_new.replace("(", "")
                                var_obj = var_obj.rename(columns={col: col_new})
                            if ")" in col:
                                col_new=col.replace(")", "")
                                while ")" in col_new:
                                    col_new=col_new.replace(")", "")
                                var_obj = var_obj.rename(columns={col: col_new})
                            if " " in col:
                                col_new=col.replace(" ", "")
                                while " " in col_new:
                                    col_new=col_new.replace(" ", "")
                                var_obj = var_obj.rename(columns={col: col_new})
                            if "%" in col:
                                col_new=col.replace("%", "")
                                while "%" in col_new:
                                    col_new=col_new.replace("%", "")
                                var_obj = var_obj.rename(columns={col: col_new})
                    except:
                        pass
                    logging.info(f"storing table... : {store_var_name}")
                    err=self.store_table3(var_name, var_obj, store_var_name)
                    if err:
                        ret_err=True
                    else:
                        logging.info(f"storing table is completed : {store_var_name}")
                except Exception as e:
                    #if "DataFrame" in var_outputs[cell_id][0][var_name][0] or "ndarray" in var_outputs[cell_id][0][var_name][0]:
                    if "DataFrame" in var_outputs[cell_id][0][var_name][0]:
                        logging.info(f"failed storing table: {store_var_name} because {e}")
                        ret_err=True

            for var_name in var_outputs[cell_id][1]:
                if var_name in var_outputs[cell_id][0]:
                    continue
                store_var_name = f"{cell_id}_{var_name}_{cleaned_nb_name}" #テーブルを保存する時の名前を生成
                try: #変数の内容をデータベースに保存
                    var_obj=pd.read_json(var_outputs[cell_id][1][var_name][1]["text"], orient="split")
                    for col in var_obj.columns:
                        if "(" in col:
                            col_new=col.replace("(", "")
                            var_obj = var_obj.rename(columns={col: col_new})
                        if ")" in col:
                            col_new=col.replace(")", "")
                            var_obj = var_obj.rename(columns={col: col_new})
                        if " " in col:
                            col_new=col.replace(" ", "")
                            var_obj = var_obj.rename(columns={col: col_new})
                        if "%" in col:
                            col_new=col.replace("%", "")
                            var_obj = var_obj.rename(columns={col: col_new})
                    logging.info(f"storing table... : {store_var_name}")
                    err=self.store_table3(var_name, var_obj, store_var_name)
                    if err:
                        ret_err=True
                    else:
                        logging.info(f"storing table is completed : {store_var_name}")
                except Exception as e:
                    if "DataFrame" in var_outputs[cell_id][1][var_name][0]:
                        logging.info(f"failed storing table: {store_var_name} because {e}")
                        ret_err=True
        
        return ret_err

    def store_cell(self, cell_id, store_cell_name, code_list):#handlerからコピー
        """
        (cell_id, code_list)のタプルをデータベースにストア．
        -> "dependen"などのあの3つのような形式で保存した方が良いのでは？
        """
        logging.info(f"Indexing cell. cell id: {cell_id}")

        try:
            try:
                #code_list = self.code.split("\\n#\\n")
                self.insert_cell2table(
                    store_cell_name, cell_id, code_list
                )
                #print(f"cell id {cell_id} : completed")
            except Exception as e:
                logging.error(
                    f"Unable to store {store_cell_name} cell_id: {cell_id}"
                    f"due to error {e}"
                )
            logging.info(f"Returning after indexing {store_cell_name}")
        except Exception as e:
            logging.error(f"Unable to store {store_cell_name} due to error {e}")


    def store_cell2(self, cell_id, store_cell_name, code_list):#handlerからコピー
        """
        (cell_id, code_list)のタプルをデータベースにストア．
        -> "dependen"などのあの3つのような形式で保存した方が良いのでは？
        """
        logging.info(f"Indexing cell. cell id: {cell_id}")

        try:
            try:
                #code_list = self.code.split("\\n#\\n")
                self.insert_cell2table2(
                    store_cell_name, cell_id, code_list
                )
                #print(f"cell id {cell_id} : completed")
            except Exception as e:
                logging.error(
                    f"Unable to store {store_cell_name} cell_id: {cell_id}"
                    f"due to error {e}"
                )
            logging.info(f"Returning after indexing {store_cell_name}")
        except Exception as e:
            logging.error(f"Unable to store {store_cell_name} due to error {e}")

    def insert_cell2table(self, table_name, cell_id, cell_code):
        """
        データベースのテーブル"table_name"にタプルをINSERTする．
        """
        #conn = self.psql_engine
        if type(cell_code) == type([]):
            cell_code="\n".join(cell_code)
        flg=True
        with self.psql_engine.connect() as conn:
            try:
                old_table = pd.read_sql_table(f"{table_name}", conn, schema=f"{config.sql.cellcode}")
                #cid.astype(type(old_table["cell_id"][0]))
                cid=cell_id
                cid=cid.astype(np.int64)
                if cid in [old_table["cell_id"]]:
                    logging.info(f"cell id: {cell_id} is already stored.")
                    flg=False
                else:
                    flg=True
            except:
                flg=True
            if flg:
                logging.info(f"storing cell...  cell id: {cell_id}")
                conn.execute(
                    f"INSERT INTO {config.sql.cellcode}.{table_name} VALUES ('{cell_id}', '{cell_code}')"
                )
            

    def insert_cell2table2(self, table_name, cell_id, cell_code):
        """
        データベースのテーブル"table_name"にタプルをINSERTする．
        """
        #conn = self.psql_engine
        if type(cell_code) == type([]):
            cell_code="\n".join(cell_code)
        flg=True
        with self.psql_engine.connect() as conn:
            try:
                old_table = pd.read_sql_table(f"{table_name}", conn, schema=f"{config.sql.cellcode}")
                #cid.astype(type(old_table["cell_id"][0]))
                cid=cell_id
                cid=cid.astype(np.int64)
                if cid in [old_table["cell_id"]]:
                    logging.info(f"cell id: {cell_id} is already stored.")
                    flg=False
                else:
                    flg=True
            except:
                flg=True
            if flg:
                #行を追加したものをcurrent_tableとする
                insert_row_set = pd.Series([cell_id, cell_code], index=old_table.columns)
                current_table=old_table.append(insert_row_set,ignore_index=True)
                print("***********")
                print("***********")
                print(old_table)
                print("***********")
                print(current_table)
                print("***********")
                print("***********")
                try:
                    current_table.to_sql(
                        name=f"rtable{table_name}",
                        con=conn,
                        schema=config.sql.cellcode, # schema name(not table definitions)
                        if_exists="replace", # 既にあった場合は置き換え
                        index=False,
                    )
                except Exception as e:
                    logging.error(f"Unable to store {table_name} due to error {e}")

    def insert_cell2table3(self, table_name, cell_code_dict):
        """
        まとめてinsert
        """
        try: # (cell_id, cell_code)のタプルをデータベースのcolumsとして初期化
            self.init_table(table_name)
        except Exception as e:
            logging.error(f"init_table: error {e}")   
        #conn = self.psql_engine
        for cid in cell_code_dict:
            if type(cell_code_dict[cid]) == type([]):
                cell_code_dict[cid]="\n".join(cell_code_dict[cid])
        flg=True
        with self.psql_engine.connect() as conn:
            try:
                old_table = pd.read_sql_table(f"{table_name}", conn, schema=f"{config.sql.cellcode}")
            except:
                pass
            if flg:
                #行を追加したものをcurrent_tableとする
                current_table=old_table
                for cid in cell_code_dict:
                    #cid2=cid.astype(np.int64)
                    #if cid2 in [old_table["cell_id"]] or cid in [old_table["cell_id"]]:
                    if cid in old_table["cell_id"].values:
                        #logging.info(f"cell id: {cell_id} is already stored.")
                        continue
                    insert_row_set = pd.Series([cid, cell_code_dict[cid]], index=old_table.columns)
                    current_table=current_table.append(insert_row_set,ignore_index=True)

                #print("***********")
                #print("***********")
                #print(old_table)
                #print("***********")
                #print(current_table)
                #print("***********")
                #print("***********")
                try:
                    current_table.to_sql(
                        name=f"{table_name}",
                        con=conn,
                        schema=config.sql.cellcode, # schema name(not table definitions)
                        if_exists="replace", # 既にあった場合は置き換え
                        index=False,
                    )
                except Exception as e:
                    logging.error(f"Unable to store {table_name} due to error {e}")

    def insert_cell2table3_2(self, table_name, cell_code_dict):
        """
        まとめてinsert
        """
        try: # (cell_id, cell_code)のタプルをデータベースのcolumsとして初期化
            self.init_table(table_name)
        except Exception as e:
            logging.error(f"init_table: error {e}")   
        #conn = self.psql_engine
        for cid in cell_code_dict:
            if type(cell_code_dict[cid]) == type([]):
                cell_code_dict[cid]="\n".join(cell_code_dict[cid])
        with self.psql_engine.connect() as conn:
            current_table=pd.DataFrame()
            for cid in cell_code_dict:
                if cell_code_dict[cid]=="":
                    continue
                insert_row_set = pd.Series([cid, cell_code_dict[cid]], index=["cell_id", "cell_code"])
                current_table=current_table.append(insert_row_set,ignore_index=True)

            try:
                current_table.to_sql(
                    name=f"{table_name}",
                    con=conn,
                    schema=config.sql.cellcode, # schema name(not table definitions)
                    if_exists="replace", # 既にあった場合は置き換え
                    index=False,
                )
            except Exception as e:
                logging.error(f"Unable to store {table_name} due to error {e}")

    def init_table(self, table_name):
        with self.psql_engine.connect() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {config.sql.cellcode}.{table_name} " 
                f"(cell_id INTEGER, cell_code TEXT);" # VARCHAR(M)は, 最大M文字数の可変長文字列の型.
            )

    def init_schema(self):
        with self.psql_engine.connect() as conn:
            conn.execute(
                f"CREATE SCHEMA IF NOT EXISTS {config.sql.cellcode};" 
            )


    def store_table2(self, var_name, output, store_table_name):#handlerからコピー
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
        #logging.info(f"Indexing new table {store_table_name}")
        conn = self.psql_engine.connect()
        #conn = self.psql_engine

        try:
            output.to_sql(
                name=f"rtable{store_table_name}",
                con=conn,
                schema=config.sql.dbs, # schema name(not 'table definitions')
                if_exists="replace", # 既にあった場合は置き換え
                index=False,
            )
            #logging.info("Base table stored")
            #**************ここまでおk
            # ここで変数名とコードのdependencyなどを保存．
            #try:
                #code_list = self.code.split("\\n#\\n")
                #self.store_prov_db_class.insert_table_model(
                    #store_table_name, var_name, self.nb_all_code
                #)
                #self.indexed.add(store_table_name)
            #except Exception as e:
                #logging.error(
                    #f"Unable to store provenance of {store_table_name} "
                    #f"due to error {e}"
                #)
            #logging.info(f"Returning after indexing {store_table_name}")
        except Exception as e:
            logging.error(f"Unable to store {store_table_name} due to error {e}")
        finally:
            conn.close()

    def store_table3(self, var_name, output, store_table_name):#handlerからコピー
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
        #logging.info(f"Indexing new table {store_table_name}")
        conn = self.psql_engine.connect()
        #conn = self.psql_engine
        err=False
        try:
            output.to_sql(
                name=f"rtable{store_table_name}",
                con=conn,
                schema=config.sql.dbs, # schema name(not 'table definitions')
                if_exists="replace", # 既にあった場合は置き換え
                index=False,
            )
        except Exception as e:
            err=True
            logging.error(f"Unable to store {store_table_name} due to error {e}")
        finally:
            conn.close()
        return err

    def store_all(self, var, code, cell_id):
        """
        実行したセルのセルとテーブルをデータベースにStoreする．
        セルはノード間のリレーションを設定する事により，ワークフローにノードとして追加することを実現している．

        self.prev_node (Node): 次にセルをノードとして追加するときにリレーションで接続するノード．
        """
        self.var=var
        self.code=code
        self.cell_id=cell_id
        logging.info(f"Indexing request: {self.var}")
        logging.info(f"Stored tables: {self.indexed}")

        cleaned_nb_name = clean_notebook_name(self.nb_name) #ノートブック名を整える
        code_list = self.code.strip("\\n#\\n").split("\\n#\\n") #codeを改行で区切ってリストにする
        store_table_name = f"{self.cell_id}_{self.var}_{cleaned_nb_name}" #テーブルを保存する時の名前を自動で生成

        if store_table_name in self.indexed: #すでに同じ名前のテーブル名が存在する時
            logging.info("Request to index is already registered.")
        elif self.var not in code_list[-1]:
            logging.info("Not a variable in the current cell.")
        else: # セルをワークフローグラフに保存. テーブルもデータベースに保存.
            logging.info(f"Starting to store {self.var}")
            success, output = self.find_variable()

            if success:
                logging.info(f"Getting value of {self.var}")
                logging.info(output.head())
                
                if cleaned_nb_name not in self.nb_cell_id_node:
                    self.nb_cell_id_node[cleaned_nb_name] = {}
                    
                # セルをデータベースに保存(ワークフローグラフへの追加)
                try:
                    # cell_id以下の番号のうち、nb_cell_id_nodeに格納されている最大のcell_idをもつノードをprev_nodeとする. なければNoneのまま
                    for cid in range(self.cell_id - 1, -1, -1): # C++でいうfor(int i=cell_id; i>-1; i--)と同じ
                        if cid in self.nb_cell_id_node[cleaned_nb_name]:
                            self.prev_node = self.nb_cell_id_node[cleaned_nb_name][cid]
                            break
                    self.prev_node = self.store_graph_db_class.add_cell( # for cell provenance tracking (セルをワークフローグラフに追加)
                        self.code,
                        self.prev_node,
                        self.var,
                        self.cell_id,
                        cleaned_nb_name,
                    )
                    # cell_idがまだnb_cell_id_nodeに格納されていなければ，その添字cell_idにprev_nodeを格納する
                    if self.cell_id not in self.nb_cell_id_node[cleaned_nb_name]:
                        self.nb_cell_id_node[cleaned_nb_name][self.cell_id] = self.prev_node
                except Exception as e:
                    logging.error(f"Unable to store in graph store due to error {e}")

                self.store_table(output, store_table_name) # テーブルをデータベースに保存

            else:
                logging.error("find variable failed!")

    def store_all_no_table(self, var, code, cell_id):
        """
        実行したセルのセルとテーブルをデータベースにStoreする．
        セルはノード間のリレーションを設定する事により，ワークフローにノードとして追加することを実現している．

        self.prev_node (Node): 次にセルをノードとして追加するときにリレーションで接続するノード．
        """
        self.var=var
        self.code=code
        self.cell_id=cell_id
        logging.info(f"Indexing request: {self.var}")
        logging.info(f"Stored tables: {self.indexed}")

        cleaned_nb_name = clean_notebook_name(self.nb_name) #ノートブック名を整える
        code_list = self.code.strip("\\n#\\n").split("\\n#\\n") #codeを改行で区切ってリストにする
        store_table_name = f"{self.cell_id}_{self.var}_{cleaned_nb_name}" #テーブルを保存する時の名前を自動で生成

        if store_table_name in self.indexed: #すでに同じ名前のテーブル名が存在する時
            logging.info("Request to index is already registered.")
        elif self.var not in code_list[-1]:
            logging.info("Not a variable in the current cell.")
        else: # セルをワークフローグラフに保存. テーブルもデータベースに保存
            if cleaned_nb_name not in self.nb_cell_id_node:
                self.nb_cell_id_node[cleaned_nb_name] = {}
            
            # セルをデータベースに保存(ワークフローグラフへの追加)
            try:
                # cell_id以下の番号のうち、nb_cell_id_nodeに格納されている最大のcell_idをもつノードをprev_nodeとする. なければNoneのまま
                for cid in range(self.cell_id - 1, -1, -1): # C++でいうfor(int i=cell_id; i>-1; i--)と同じ
                    if cid in self.nb_cell_id_node[cleaned_nb_name]:
                        self.prev_node = self.nb_cell_id_node[cleaned_nb_name][cid]
                        break
                    self.prev_node = self.store_graph_db_class.add_cell_no_table( # for cell provenance tracking (セルをワークフローグラフに追加)
                        self.code,
                        self.prev_node,
                        self.var,
                        self.cell_id,
                        cleaned_nb_name,
                    )
                    # cell_idがまだnb_cell_id_nodeに格納されていなければ，その添字cell_idにprev_nodeを格納する
                    if self.cell_id not in self.nb_cell_id_node[cleaned_nb_name]:
                        self.nb_cell_id_node[cleaned_nb_name][self.cell_id] = self.prev_node
            except Exception as e:
                logging.error(f"Unable to store in graph store due to error {e}")


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
            # ここで変数名とコードのdependencyなどを保存．
            try:
                code_list = self.code.split("\\n#\\n")
                self.store_prov_db_class.insert_table_model(
                    store_table_name, self.var, code_list
                )
                self.indexed.add(store_table_name)
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
