import logging
import timeit

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

from module2.config import config
from module2.utils.cost_func import compute_table_size


class SeparateStorage:
    def __init__(self, dbname, time_flag=False):
        self.dbname = dbname
        self.__connect2db_init()
        self.eng = self.__connect2db()
        self.time_flag = time_flag
        self.variable = []
        self.update_time = 0

    def __connect2db(self):
        """
        PostgreSQLでデータベースにアクセス．

        Returns:
            engine.connect()
        """
        engine = create_engine(
            f"postgresql://{config.sql.name}:{config.sql.password}@localhost/{self.dbname}"
        )
        return engine.connect()

    def __connect2db_init(self):
        """
        FIXME: We are catching the exceptions but only logging. We do not
              reraise the exception. The code will continue to run without a database
              connection.
        """

        conn_string = (
            f"host='localhost' dbname='{self.dbname}' "
            f"user='{config.sql.name}' password='{config.sql.password}'"
        )

        logging.info(f"Connecting to database\n	->{conn_string}")

        try:
            # conn.cursor will return a cursor object, you can use this cursor to perform queries
            conn = psycopg2.connect(conn_string)
            logging.info("Connecting Database Succeeded!\n")
            cursor = conn.cursor()
            query1 = "DROP SCHEMA IF EXISTS rowstore CASCADE;"
            query2 = "CREATE SCHEMA rowstore;"

            try:
                cursor.execute(query1)
                conn.commit()
            except Exception as e:
                logging.error(f"Drop schema failed due to error {e}")
            try:
                cursor.execute(query2)
                conn.commit()
            except Exception as e:
                logging.error(f"Creation of schema failed due to error {e}")
            cursor.close()
            conn.close()
        except Exception as e:
            logging.info(f"Connection to database failed due to error {e}")

    def insert_table_separately(self, idi, new_table):
        """
        すでに保存されていても別でテーブルをデータベースに保存する．
        データベースのテーブルを更新するときは`update_data`を利用．
        self.variableも更新(引数idiを追加)する．
        self.time_flagがTrueならかかった時間をログに書き込む．
        """
        if self.time_flag:
            start_time = timeit.default_timer()

        new_table.to_sql(
            name=f"rtable{idi}",
            con=self.eng,
            schema="rowstore",
            if_exists="fail",
            index=False,
        )
        self.variable.append(idi)

        if self.time_flag:
            end_time = timeit.default_timer()
            logging.info(end_time - start_time)

    def query_storage_size(self):
        """
        データベース内の全てのテーブルサイズの合計(byte)を計算する．

        Returns:
            float: データベース内の全てのテーブルサイズの合計(byte)
        """
        eng = self.__connect2db()
        mediate_tables = eng.execute( # テーブル名をリストに格納する．
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'rowstore';"
        )
        table_name = []
        storage_number = [] # テーブルサイズのリスト
        for row in mediate_tables:
            table_name.append(("rowstore", row[0]))
        for sch, tn in table_name: 
            try:
                table = pd.read_sql_table(tn, eng, schema=sch) #テーブルを読み込む
            except:
                logging.error(tn)
                continue
            storage_number.append(compute_table_size(table)) # それぞれのテーブルのサイズ(float)をリストに追加
        eng.close()
        return float(sum(storage_number))

    def close_dbconnection(self):
        self.eng.close()

    def update_data(self, idi, new_table, vid):
        """
        データベースのテーブルを更新する．データベースに更新対象のテーブルが存在しなければ作成する．
        データベースにテーブルが存在しても新しく別に作成したいときは`insert_table_separately`を利用．
        self.upgrade_timeにこの関数の実行にかかった時間を格納．
        vidはversion id?
        """
        start_time = timeit.default_timer() # 開始時間
        self.eng = self.__connect2db()
        nflg = True #すでにテーブルがあるときTrue
        try:
            old_table = pd.read_sql_table(f"rtable{idi}", self.eng, schema="rowstore")
        except:
            old_table = None
            nflg = False

        if not nflg: # データベースに`new_table`と一致するテーブルが無いとき
            new_table.to_sql( #DataFrameをSQLの形に変換
                name=f"rtable{idi}_{vid}",
                con=self.eng,
                index=False,
                schema="rowstore",
                if_exists="replace",
            )
        else:
            if not new_table.equals(old_table): #new_tableがold_tableと異なった場合は更新
                new_table.to_sql(
                    name=f"rtable{idi}_{vid}",
                    con=self.eng,
                    index=False,
                    schema="rowstore",
                    if_exists="replace",
                )
        end_time = timeit.default_timer() #終了時間
        self.update_time = self.update_time + end_time - start_time #かかった時間
        self.eng.close()
